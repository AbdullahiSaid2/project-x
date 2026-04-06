from dataclasses import dataclass, asdict


@dataclass
class StrategySchema:
    family: str
    name: str
    description: str

    def to_dict(self):
        return asdict(self)


def schema_from_dict(raw, source_idea=""):
    return StrategySchema(
        family=raw.get("family", "generic"),
        name=raw.get("name", "Strategy"),
        description=raw.get("description", source_idea),
    )