FROM rust:1.78-slim AS builder
WORKDIR /build
COPY . .
RUN cargo build --release --bin strategos

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /build/target/release/strategos .
COPY knowledge/ knowledge/
ENV DISPATCHER_URL=http://dispatcher:8080/v1/chat/completions
ENV REASONER_URL=http://reasoner:8081/v1/chat/completions
CMD ["./strategos"]
