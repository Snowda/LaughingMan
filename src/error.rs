//! Crate error type: a `thiserror` enum plus a `Context` trait for anyhow-style `.context()`.

use std::fmt;

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error("{message}: {source}")]
    Context { message: String, source: BoxError },
}

impl Error {
    pub fn msg(message: impl fmt::Display) -> Self {
        Error::Message(message.to_string())
    }
}

// `Debug` reuses `Display` so `fn main() -> Result<()>` prints the full cause chain, not fields.
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// anyhow-style `.context()` / `.with_context()` on `Result`/`Option`, backed by [`Error`].
pub trait Context<T> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T>;
    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T>;
}

impl<T, E: std::error::Error + Send + Sync + 'static> Context<T> for Result<T, E> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T> {
        self.map_err(|source| Error::Context { message: context.to_string(), source: Box::new(source) })
    }
    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.map_err(|source| Error::Context { message: f().to_string(), source: Box::new(source) })
    }
}

impl<T> Context<T> for Option<T> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T> {
        self.ok_or_else(|| Error::Message(context.to_string()))
    }
    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.ok_or_else(|| Error::Message(f().to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{Context, Error};
    use std::cell::Cell;
    use std::io::Error as IoError;

    fn io(msg: &str) -> IoError {
        IoError::other(msg)
    }

    #[test]
    fn msg_renders_the_display_and_debug_mirrors_it() {
        let e: Error = Error::msg(42);
        assert_eq!(e.to_string(), "42", "msg accepts any Display");
        assert_eq!(format!("{e:?}"), "42", "Debug reuses Display, not the field-wise derive");
    }

    #[test]
    fn result_context_chains_the_cause() {
        let err: Option<String> = Result::<(), IoError>::Err(io("disk gone"))
            .context("saving atlas")
            .err()
            .map(|e| e.to_string());
        assert_eq!(err.as_deref(), Some("saving atlas: disk gone"));
    }

    #[test]
    fn result_with_context_is_lazy_and_chains() {
        let calls: Cell<i32> = Cell::new(0);
        let ok: Result<i32, Error> = Result::<i32, IoError>::Ok(9).with_context(|| {
            calls.set(calls.get() + 1);
            "unused"
        });
        assert_eq!(ok.ok(), Some(9), "Ok passes through untouched");
        assert_eq!(calls.get(), 0, "message closure never runs on the Ok path");

        let err: Option<String> = Result::<(), IoError>::Err(io("eof"))
            .with_context(|| format!("frame {}", 7))
            .err()
            .map(|e| e.to_string());
        assert_eq!(err.as_deref(), Some("frame 7: eof"));
    }

    #[test]
    fn context_exposes_the_typed_cause_via_source() {
        use std::error::Error as _;
        let err: Option<Error> = Result::<(), IoError>::Err(io("disk gone")).context("saving atlas").err();
        let cause: Option<&IoError> = err.as_ref().and_then(|e| e.source()).and_then(|s| s.downcast_ref::<IoError>());
        assert_eq!(cause.map(ToString::to_string).as_deref(), Some("disk gone"), "cause kept as the original IoError");
    }

    #[test]
    fn message_has_no_source() {
        use std::error::Error as _;
        assert!(Error::msg("bare").source().is_none(), "a bare message wraps no cause");
        let opt: Option<Error> = None::<()>.context("missing").err();
        assert!(opt.is_some_and(|e| e.source().is_none()), "Option context has no underlying cause");
    }

    #[test]
    fn nested_context_renders_the_whole_chain() {
        let inner: Result<(), Error> = Result::<(), IoError>::Err(io("eof")).context("reading frame");
        let outer: Option<String> = inner.context("probing camera").err().map(|e| format!("{e:?}"));
        assert_eq!(outer.as_deref(), Some("probing camera: reading frame: eof"));
    }

    #[test]
    fn option_context_builds_a_bare_message() {
        assert_eq!(Some(5).context("unused").ok(), Some(5));
        let err: Option<String> = None::<i32>.context("missing device").err().map(|e| e.to_string());
        assert_eq!(err.as_deref(), Some("missing device"));
    }

    #[test]
    fn option_with_context_is_lazy() {
        let calls: Cell<i32> = Cell::new(0);
        let ok: Result<i32, Error> = Some(1).with_context(|| {
            calls.set(calls.get() + 1);
            "unused"
        });
        assert_eq!(ok.ok(), Some(1));
        assert_eq!(calls.get(), 0, "closure skipped when the Option is Some");

        let err: Option<String> = None::<i32>.with_context(|| "lazy missing").err().map(|e| e.to_string());
        assert_eq!(err.as_deref(), Some("lazy missing"));
    }
}
