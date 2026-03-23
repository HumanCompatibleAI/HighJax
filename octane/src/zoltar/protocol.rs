//! Zoltar JSON protocol: request and response types.

use serde::{Deserialize, Serialize};

/// A zoltar request from an external client.
#[derive(Debug, Deserialize)]
#[serde(tag = "cmd")]
pub enum ZoltarRequest {
    #[serde(rename = "genco")]
    Genco,
    #[serde(rename = "pane")]
    Pane {
        pane: String,
        query: String,
        #[serde(default)]
        path: Option<String>,
    },
    #[serde(rename = "press")]
    Press { keys: Vec<String> },
    #[serde(rename = "navigate")]
    Navigate {
        #[serde(default)]
        epoch: Option<u64>,
        #[serde(default)]
        episode: Option<u64>,
        #[serde(default)]
        timestep: Option<u64>,
    },
    #[serde(rename = "ping")]
    Ping,
}

/// A zoltar response sent back to the client.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ZoltarResponse {
    /// Arbitrary JSON value (used for genco, pane data, etc.).
    Json(serde_json::Value),
    /// Error response.
    Error { error: String },
}

impl ZoltarResponse {
    pub fn error(msg: impl Into<String>) -> Self {
        Self::Error { error: msg.into() }
    }

    pub fn pong() -> Self {
        Self::Json(serde_json::json!({"pong": true}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_genco() {
        let req: ZoltarRequest = serde_json::from_str(r#"{"cmd":"genco"}"#).unwrap();
        assert!(matches!(req, ZoltarRequest::Genco));
    }

    #[test]
    fn test_parse_ping() {
        let req: ZoltarRequest = serde_json::from_str(r#"{"cmd":"ping"}"#).unwrap();
        assert!(matches!(req, ZoltarRequest::Ping));
    }

    #[test]
    fn test_parse_pane() {
        let req: ZoltarRequest = serde_json::from_str(
            r#"{"cmd":"pane","pane":"metrics","query":"data"}"#
        ).unwrap();
        match req {
            ZoltarRequest::Pane { pane, query, path } => {
                assert_eq!(pane, "metrics");
                assert_eq!(query, "data");
                assert!(path.is_none());
            }
            _ => panic!("expected Pane"),
        }
    }

    #[test]
    fn test_parse_press() {
        let req: ZoltarRequest = serde_json::from_str(
            r#"{"cmd":"press","keys":["g","1"]}"#
        ).unwrap();
        match req {
            ZoltarRequest::Press { keys } => {
                assert_eq!(keys, vec!["g", "1"]);
            }
            _ => panic!("expected Press"),
        }
    }

    #[test]
    fn test_parse_navigate() {
        let req: ZoltarRequest = serde_json::from_str(
            r#"{"cmd":"navigate","epoch":200,"episode":5}"#
        ).unwrap();
        match req {
            ZoltarRequest::Navigate { epoch, episode, timestep } => {
                assert_eq!(epoch, Some(200));
                assert_eq!(episode, Some(5));
                assert!(timestep.is_none());
            }
            _ => panic!("expected Navigate"),
        }
    }

    #[test]
    fn test_serialize_pong() {
        let resp = ZoltarResponse::pong();
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("pong"));
    }

    #[test]
    fn test_serialize_error() {
        let resp = ZoltarResponse::error("something broke");
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("something broke"));
    }
}
