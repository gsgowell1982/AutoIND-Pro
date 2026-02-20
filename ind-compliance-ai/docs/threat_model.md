# Threat Model

## Assets to protect

- Trade secrets in CMC and preclinical documents
- Regulatory strategy details and submission timelines
- Internal analysis traces and generated risk explanations

## Threats

- Data exfiltration through external model endpoints
- Unauthorized access to vector indexes and parsed artifacts
- Prompt injection against explanation and retrieval layers

## Security baseline

- Private deployment by default
- Strict network egress controls
- At-rest encryption for data and indexes
- Role-based access control and audit logging
- Signed version records for rules and regulation corpora
