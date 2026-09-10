//! `ResponseChunk` -> SMG `GenerateResponse`.
//!
//! Two properties of the producer decide the whole module.
//!
//! **Our `output_ids` are cumulative; SMG's chunk contract is incremental.**
//! `--incremental-streaming-output` defaults false, so every chunk carries the
//! entire sequence so far. Passing that through makes SMG concatenate
//! `"a" "ab" "abc"` while `completion_tokens` stays exactly right, so nothing
//! anywhere reports a problem. `StreamTranslator` differences it, and asserts
//! the prefix invariant so a misdetected mode is loud instead.
//!
//! **Every `meta_info` value is `json.dumps`-encoded.** `finish_reason` is the
//! literal four-character `null` while streaming, `id` arrives with quote
//! characters inside it, and `finish_reason` at the end is a JSON object. Every
//! read here goes through `serde_json`, and `contains_key("finish_reason")` is
//! never a completion test — the `ResponseChunk` variant is.

use std::collections::HashMap;

use tonic::Status;

use crate::bridge::ResponseData;
use crate::smg_proto;

/// Whether the engine is emitting cumulative or incremental `output_ids`.
///
/// Read once at startup from `server_args.incremental_streaming_output`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// `--incremental-streaming-output` false, the default.
    Cumulative,
    Incremental,
}

/// One generation stream's worth of state.
///
/// One translator per `index`, and `index` is always 0: `runtime.v1` has no
/// per-sample discriminator, so `request_map` rejects `n > 1`. Held as state
/// rather than written at each construction site so the two files disagree
/// loudly rather than drifting.
pub struct StreamTranslator {
    mode: StreamMode,
    index: u32,
    /// Every token delivered so far. `GenerateComplete.output_ids` must be the
    /// whole sequence: non-streaming SMG reconstructs the answer from it alone.
    full: Vec<u32>,
    logprobs: OutputLogprobAccumulator,
}

#[derive(Default)]
struct OutputLogprobAccumulator {
    token_logprobs: Vec<f32>,
    token_ids: Vec<i32>,
    top_logprobs: Vec<smg_proto::TopLogProbs>,
}

impl StreamTranslator {
    pub fn new(mode: StreamMode) -> Self {
        Self {
            mode,
            index: 0,
            full: Vec::new(),
            logprobs: OutputLogprobAccumulator::default(),
        }
    }

    /// A non-terminal chunk. `None` means "emit nothing": a chunk that adds no
    /// tokens would still cost SMG's Harmony path one completion token, which
    /// increments per chunk rather than per token.
    pub fn on_data(
        &mut self,
        data: &ResponseData,
    ) -> Result<Option<smg_proto::GenerateResponse>, Status> {
        let delta = self.take_delta(data)?;
        let logprob_delta = self.take_logprob_delta(&data.meta_info)?;
        if delta.is_empty() {
            return Ok(None);
        }
        Ok(Some(self.chunk_message(
            delta,
            logprob_delta,
            &data.meta_info,
        )?))
    }

    /// The terminal chunk, which yields up to two messages.
    ///
    /// Our bridge's `Finished` carries the whole cumulative sequence, so the
    /// tokens produced since the last `Data` have never been sent as a chunk.
    /// Streaming SMG builds its text purely by accumulating chunks and takes
    /// only the *last* element of `complete.output_ids`, so emitting `complete`
    /// alone truncates the stream by the final delta while `completion_tokens`
    /// stays correct. `complete.output_ids` therefore deliberately re-sends
    /// tokens already delivered as chunks; that is the contract.
    pub fn on_finished(
        &mut self,
        data: &ResponseData,
    ) -> Result<Vec<smg_proto::GenerateResponse>, Status> {
        let delta = self.take_delta(data)?;
        let logprob_delta = self.take_logprob_delta(&data.meta_info)?;

        let mut messages = Vec::with_capacity(2);
        if !delta.is_empty() {
            messages.push(self.chunk_message(delta, logprob_delta, &data.meta_info)?);
        }
        messages.push(self.complete_message(&data.meta_info)?);
        Ok(messages)
    }

    /// The new tokens in this chunk, advancing `full`.
    fn take_delta(&mut self, data: &ResponseData) -> Result<Vec<u32>, Status> {
        let incoming = to_output_ids(data.output_ids.as_deref().unwrap_or(&[]))?;

        match self.mode {
            StreamMode::Cumulative => {
                if incoming.len() < self.full.len() || incoming[..self.full.len()] != self.full[..]
                {
                    return Err(Status::internal(
                        "cumulative output_ids are not a prefix extension of the previous chunk; \
                         the engine's incremental_streaming_output was misdetected",
                    ));
                }
                let delta = incoming[self.full.len()..].to_vec();
                self.full = incoming;
                Ok(delta)
            }
            StreamMode::Incremental => {
                self.full.extend_from_slice(&incoming);
                Ok(incoming)
            }
        }
    }

    /// The new output logprobs, advancing the accumulator. Sliced exactly like
    /// `output_ids`: in cumulative mode `meta_info["output_token_logprobs"]` is
    /// the whole list every chunk, and SMG's chunk field is incremental.
    fn take_logprob_delta(
        &mut self,
        meta_info: &HashMap<String, String>,
    ) -> Result<Option<smg_proto::OutputLogProbs>, Status> {
        let Some(entries) = parse_output_token_logprobs(meta_info)? else {
            return Ok(None);
        };
        let tops = parse_output_top_logprobs(meta_info)?;

        let already_sent = match self.mode {
            StreamMode::Cumulative => self.logprobs.token_ids.len(),
            StreamMode::Incremental => 0,
        };
        if entries.len() < already_sent {
            return Err(Status::internal(
                "cumulative output_token_logprobs shrank between chunks",
            ));
        }

        let mut delta = smg_proto::OutputLogProbs::default();
        for (logprob, token_id) in &entries[already_sent..] {
            delta.token_logprobs.push(*logprob);
            delta.token_ids.push(*token_id);
        }
        if let Some(tops) = tops {
            if tops.len() < already_sent {
                return Err(Status::internal(
                    "cumulative output_top_logprobs shrank between chunks",
                ));
            }
            delta.top_logprobs.extend_from_slice(&tops[already_sent..]);
        }

        self.logprobs
            .token_logprobs
            .extend_from_slice(&delta.token_logprobs);
        self.logprobs.token_ids.extend_from_slice(&delta.token_ids);
        self.logprobs
            .top_logprobs
            .extend_from_slice(&delta.top_logprobs);

        Ok(Some(delta))
    }

    fn chunk_message(
        &self,
        token_ids: Vec<u32>,
        output_logprobs: Option<smg_proto::OutputLogProbs>,
        meta_info: &HashMap<String, String>,
    ) -> Result<smg_proto::GenerateResponse, Status> {
        let chunk = smg_proto::GenerateStreamChunk {
            token_ids,
            prompt_tokens: meta_count(meta_info, "prompt_tokens")?,
            completion_tokens: meta_count(meta_info, "completion_tokens")?,
            cached_tokens: meta_count(meta_info, "cached_tokens")?,
            output_logprobs,
            // SMG hard-codes None for hidden states and never reads
            // `input_logprobs` off the chunk variant.
            hidden_states: Vec::new(),
            input_logprobs: None,
            index: self.index,
        };
        Ok(smg_proto::GenerateResponse {
            request_id: String::new(),
            response: Some(smg_proto::generate_response::Response::Chunk(chunk)),
        })
    }

    fn complete_message(
        &self,
        meta_info: &HashMap<String, String>,
    ) -> Result<smg_proto::GenerateResponse, Status> {
        let (finish_reason, matched_stop) = parse_finish_reason(meta_info);
        let complete = smg_proto::GenerateComplete {
            output_ids: self.full.clone(),
            finish_reason,
            // These three, not the chunk counters, are what SMG renders as
            // usage: a sentinel capture showed a client reporting the values
            // from here and ignoring the ones on the chunks.
            prompt_tokens: meta_count(meta_info, "prompt_tokens")?,
            completion_tokens: meta_count(meta_info, "completion_tokens")?,
            cached_tokens: meta_count(meta_info, "cached_tokens")?,
            output_logprobs: self.accumulated_logprobs(),
            all_hidden_states: Vec::new(),
            matched_stop,
            input_logprobs: parse_input_logprobs(meta_info)?,
            index: self.index,
        };
        Ok(smg_proto::GenerateResponse {
            request_id: String::new(),
            response: Some(smg_proto::generate_response::Response::Complete(complete)),
        })
    }

    fn accumulated_logprobs(&self) -> Option<smg_proto::OutputLogProbs> {
        if self.logprobs.token_ids.is_empty() {
            return None;
        }
        Some(smg_proto::OutputLogProbs {
            token_logprobs: self.logprobs.token_logprobs.clone(),
            token_ids: self.logprobs.token_ids.clone(),
            top_logprobs: self.logprobs.top_logprobs.clone(),
        })
    }
}

/// The two-message error frame.
///
/// A 1.0.0 client reads `GenerateError.message` off field 4 and stops; a 1.12.0
/// client, for which field 4 is `reserved`, skips it and takes the status.
/// Neither hangs.
pub fn error_messages(message: String) -> (smg_proto::GenerateResponse, Status) {
    let response = smg_proto::GenerateResponse {
        request_id: String::new(),
        response: Some(smg_proto::generate_response::Response::Error(
            smg_proto::GenerateError {
                message: message.clone(),
                // SMG has no accessor for either of these.
                http_status_code: String::new(),
                details: String::new(),
            },
        )),
    };
    (response, Status::internal(message))
}

/// Stamp SMG's own request id on a response for log correlation.
///
/// Never `meta_info["id"]`: that value is `json.dumps`-encoded, so it arrives
/// with literal quote characters in it.
pub fn with_request_id(
    mut response: smg_proto::GenerateResponse,
    request_id: &str,
) -> smg_proto::GenerateResponse {
    response.request_id = request_id.to_string();
    response
}

/// Build an `EmbedResponse` in **both** wire layouts.
///
/// smg-grpc-client 1.0.0 reads `complete.embedding` off the oneof at field 2;
/// 1.12.0 reserves 1-3 and reads flat fields 4-6. The numbers are disjoint, so
/// each client skips the other's as unknown. Emitting only one layout hands the
/// other client an empty embedding and no error at all — confirmed as an
/// `HTTP 500 embedding_no_response` when only the 1.12.0 layout was sent.
pub fn embed_response(
    request_id: &str,
    data: &ResponseData,
) -> Result<smg_proto::EmbedResponse, Status> {
    // `extract::<Vec<f32>>()` in the bridge yields None for sparse
    // (Dict[int, float]) and multi-vector poolers, which would otherwise become
    // a zero-length embedding with no error and a cosine similarity of 0
    // everywhere downstream.
    let embedding = data
        .embedding
        .clone()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| {
            Status::internal(
                "embedding is empty; the model's pooler is probably not dense, which \
             the bridge cannot extract",
            )
        })?;
    let prompt_tokens = meta_count(&data.meta_info, "prompt_tokens")?;
    let embedding_dim = embedding.len() as i32;

    Ok(smg_proto::EmbedResponse {
        request_id: request_id.to_string(),
        response: Some(smg_proto::embed_response::Response::Complete(
            smg_proto::EmbedComplete {
                embedding: embedding.clone(),
                prompt_tokens,
                // Not emitted for embeddings: tokenizer_manager guards the
                // meta_info update behind `not isinstance(BatchEmbeddingOutput)`.
                // 0 here means unknown, not "no cache hits". SMG never reads it.
                cached_tokens: 0,
                embedding_dim,
                // Our embed path returns one vector and SMG emits exactly one
                // EmbeddingObject regardless.
                batch_embeddings: Vec::new(),
            },
        )),
        embedding_dim_v112: embedding_dim as u32,
        embedding_v112: embedding,
        prompt_tokens_v112: prompt_tokens.max(0) as u32,
    })
}

pub fn embed_error(request_id: &str, message: String) -> smg_proto::EmbedResponse {
    smg_proto::EmbedResponse {
        request_id: request_id.to_string(),
        response: Some(smg_proto::embed_response::Response::Error(
            smg_proto::EmbedError {
                message,
                // SMG reads neither.
                code: String::new(),
                details: String::new(),
            },
        )),
        ..Default::default()
    }
}

fn to_output_ids(ids: &[i32]) -> Result<Vec<u32>, Status> {
    ids.iter()
        .map(|&id| {
            u32::try_from(id)
                .map_err(|_| Status::internal(format!("engine emitted negative token id {}", id)))
        })
        .collect()
}

/// A `meta_info` counter. Absent means 0 — `cached_tokens` is genuinely not
/// emitted for embeddings — but present-and-not-an-integer means the producer's
/// encoding changed, which must not be rounded into a plausible number.
fn meta_count(meta_info: &HashMap<String, String>, key: &str) -> Result<i32, Status> {
    let Some(raw) = meta_info.get(key) else {
        return Ok(0);
    };
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .as_ref()
        .and_then(serde_json::Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(|| Status::internal(format!("meta_info[\"{}\"] is not an int32: {}", key, raw)))
}

fn meta_json(meta_info: &HashMap<String, String>, key: &str) -> Option<serde_json::Value> {
    let raw = meta_info.get(key)?;
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(serde_json::Value::Null) => None,
        Ok(value) => Some(value),
        Err(err) => {
            tracing::warn!("meta_info[\"{}\"] is not JSON ({}): {}", key, err, raw);
            None
        }
    }
}

type MatchedStop = smg_proto::generate_complete::MatchedStop;

/// `meta_info["finish_reason"]` -> the OpenAI-compatible string plus the
/// `matched_stop` arm.
///
/// While streaming the value is JSON `null`, whose `json.dumps` encoding is the
/// four characters `null`: truthy to `contains_key` and non-empty as a String,
/// so treating its presence as completion would end the stream after one token.
/// This function is only ever called on the terminal frame.
fn parse_finish_reason(meta_info: &HashMap<String, String>) -> (String, Option<MatchedStop>) {
    let Some(reason) = meta_json(meta_info, "finish_reason") else {
        tracing::warn!("terminal chunk has no finish_reason; sending it empty");
        return (String::new(), None);
    };

    let reason_type = reason
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string();

    let matched = match reason.get("matched") {
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
            .map(MatchedStop::MatchedTokenId),
        Some(serde_json::Value::String(s)) => Some(MatchedStop::MatchedStopStr(s.clone())),
        // FINISH_MATCHED_TOKEN accepts a list, and a single uint32 cannot carry
        // it. Dropping is honest; matched[0] would name the wrong stop token.
        Some(serde_json::Value::Array(_)) => {
            tracing::warn!("finish_reason.matched is a list; omitting matched_stop");
            None
        }
        _ => None,
    };

    (reason_type, matched)
}

/// `[logprob, token_id, text_or_null]` triples -> `(logprob, token_id)`.
fn parse_output_token_logprobs(
    meta_info: &HashMap<String, String>,
) -> Result<Option<Vec<(f32, i32)>>, Status> {
    let Some(value) = meta_json(meta_info, "output_token_logprobs") else {
        return Ok(None);
    };
    let entries = value.as_array().ok_or_else(|| {
        Status::internal("meta_info[\"output_token_logprobs\"] is not a list of triples")
    })?;

    entries
        .iter()
        .map(|entry| {
            let triple = entry.as_array().ok_or_else(|| {
                Status::internal(format!(
                    "output_token_logprobs entry is not a triple: {}",
                    entry
                ))
            })?;
            let logprob = triple
                .first()
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    Status::internal(format!(
                        "output_token_logprobs entry has no logprob: {}",
                        entry
                    ))
                })?;
            let token_id = triple
                .get(1)
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| {
                    Status::internal(format!(
                        "output_token_logprobs entry has no token id: {}",
                        entry
                    ))
                })?;
            Ok((logprob as f32, token_id as i32))
        })
        .collect::<Result<Vec<_>, Status>>()
        .map(Some)
}

/// A list of per-position triple lists. `detokenize_top_logprobs_tokens`
/// returns `None` for a position with no top-k, so entries can be JSON null.
fn parse_output_top_logprobs(
    meta_info: &HashMap<String, String>,
) -> Result<Option<Vec<smg_proto::TopLogProbs>>, Status> {
    let Some(value) = meta_json(meta_info, "output_top_logprobs") else {
        return Ok(None);
    };
    let positions = value.as_array().ok_or_else(|| {
        Status::internal("meta_info[\"output_top_logprobs\"] is not a list of positions")
    })?;

    let mut parsed = Vec::with_capacity(positions.len());
    for position in positions {
        let mut top = smg_proto::TopLogProbs::default();
        if let Some(entries) = position.as_array() {
            for entry in entries {
                let triple = entry.as_array().ok_or_else(|| {
                    Status::internal(format!(
                        "output_top_logprobs entry is not a triple: {}",
                        entry
                    ))
                })?;
                top.values.push(
                    triple
                        .first()
                        .and_then(serde_json::Value::as_f64)
                        .ok_or_else(|| {
                            Status::internal("output_top_logprobs entry has no logprob")
                        })? as f32,
                );
                top.token_ids.push(
                    triple
                        .get(1)
                        .and_then(serde_json::Value::as_i64)
                        .ok_or_else(|| {
                            Status::internal("output_top_logprobs entry has no token id")
                        })? as i32,
                );
            }
        }
        parsed.push(top);
    }
    Ok(Some(parsed))
}

/// Prompt logprobs. The first entry's logprob is JSON null and must stay
/// `None`: a logprob of exactly 0.0 means probability 1, a confident lie.
///
/// `InputLogProbs.top_logprobs` is left empty because SMG never reads it.
fn parse_input_logprobs(
    meta_info: &HashMap<String, String>,
) -> Result<Option<smg_proto::InputLogProbs>, Status> {
    let Some(value) = meta_json(meta_info, "input_token_logprobs") else {
        return Ok(None);
    };
    let entries = value.as_array().ok_or_else(|| {
        Status::internal("meta_info[\"input_token_logprobs\"] is not a list of triples")
    })?;

    let mut parsed = smg_proto::InputLogProbs::default();
    for entry in entries {
        let triple = entry.as_array().ok_or_else(|| {
            Status::internal(format!(
                "input_token_logprobs entry is not a triple: {}",
                entry
            ))
        })?;
        parsed.token_logprobs.push(smg_proto::InputTokenLogProb {
            value: triple
                .first()
                .and_then(serde_json::Value::as_f64)
                .map(|logprob| logprob as f32),
        });
        parsed.token_ids.push(
            triple
                .get(1)
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| Status::internal("input_token_logprobs entry has no token id"))?
                as i32,
        );
    }
    Ok(Some(parsed))
}

#[cfg(test)]
mod tests;
