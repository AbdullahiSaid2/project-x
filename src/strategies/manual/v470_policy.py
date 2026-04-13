from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class V470Policy:
    version: str = "v470"
    account_size: float = 50000.0
    account_label: str = "Apex 50k"
    fixed_contracts: int = 10
    contract_symbol: str = "MNQ"
    dollar_per_point_per_contract: float = 2.0
    allowed_setup_tiers: tuple = ("A",)
    allowed_setup_types: tuple = ("LONDON_CONTINUATION", "NYPM_CONTINUATION")
    allowed_bridge_types: tuple = ("iFVG", "C2C3", "MSS", "CISD")
    allowed_entry_variants: tuple = ("PULLBACK_1M",)
    min_planned_rr: float = 5.0
    min_profit_target_dollars: float = 500.0
    prop_daily_loss_limit: float = -1000.0
    prop_max_drawdown_limit: float = -2000.0

    @property
    def min_profit_target_points_per_contract(self) -> float:
        return self.min_profit_target_dollars / (self.fixed_contracts * self.dollar_per_point_per_contract)

    def to_dict(self):
        d = asdict(self)
        d["min_profit_target_points_per_contract"] = self.min_profit_target_points_per_contract
        return d

POLICY = V470Policy()
