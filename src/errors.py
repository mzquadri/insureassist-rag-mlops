"""
Typed failures for the two external stages of the pipeline.

The service depends on things it does not control: a vector database, an embedding model
that may need downloading, and a generator process. When one of those is missing the
request has not gone *wrong* - the service is temporarily unable to serve it. That is a
503, not a 500, and the distinction matters to anything upstream deciding whether to retry.

Only the boundaries raise these. Nothing inside the pipeline should catch them.
"""


class DependencyUnavailable(RuntimeError):
    """A backing service this request needed was not reachable."""

    #: Short machine-readable name, surfaced in the error body so a caller can tell which
    #: half of the pipeline is down without parsing prose.
    dependency = "dependency"


class RetrievalUnavailable(DependencyUnavailable):
    """The embedder or the vector store could not be reached or loaded."""

    dependency = "retrieval"


class GenerationUnavailable(DependencyUnavailable):
    """The text generator could not be reached."""

    dependency = "generator"
