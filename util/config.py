LOCAL_VALIDATOR_URL = "http://localhost:8888/?out=json"
# requires: docker run -p 8888:8888 ghcr.io/validator/validator:latest --port 8888
#
# The cloud validator (validator.w3.org) is intentionally not used: mixing cloud
# and local validators across a comparison could shift error counts for reasons
# unrelated to the model being tested (different validator versions, rate
# limiting, network flakiness). Pinning to one local, version-controlled
# validator keeps that variable constant across every run.
