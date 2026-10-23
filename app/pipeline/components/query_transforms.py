from app.pipeline.base import Component, register


class QueryTransform(Component):
    step = "query_transform"

    def transform(self, question: str) -> list[str]:
        """Returns the queries to retrieve with. Results for all of them are merged."""
        raise NotImplementedError


@register
class NoQueryTransform(QueryTransform):
    name = "none"
    label = "None"
    library = "built-in"
    description = "Use the question as written."

    def transform(self, question: str) -> list[str]:
        return [question]
