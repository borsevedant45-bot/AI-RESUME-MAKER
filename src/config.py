import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    fallback_model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 256
    embedding_dim: int = 384


@dataclass
class LLMConfig:
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 4000
    explanation_batch_size: int = 4
    explainer_model: str = "llama-3.3-70b-versatile"
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class RetrievalConfig:
    top_n_shortlist: int = 500
    top_n_output: int = 20


@dataclass
class ScoringWeights:
    semantic: float = 0.35
    trajectory: float = 0.25
    stability: float = 0.15
    platform: float = 0.20
    cert_bonus_multiplier: float = 0.05


@dataclass
class SkillStrengthConfig:
    proficiency_weight: float = 0.50
    duration_weight: float = 0.35
    endorsement_weight: float = 0.15
    max_duration_years: float = 5.0
    max_endorsements: int = 50


@dataclass
class StretchReadinessConfig:
    min_promotion_rate: float = 0.5
    min_experience_years: float = 5.0
    fit_override_value: float = 0.75


@dataclass
class TrajectoryConfig:
    seniority_levels: dict = field(default_factory=lambda: {
        "intern": 0.1, "trainee": 0.1, "junior": 0.2, "associate": 0.2,
        "entry": 0.2, "graduate": 0.2, "mid": 0.5, "senior": 0.75,
        "lead": 0.75, "manager": 0.75, "principal": 1.0, "staff": 1.0,
        "director": 1.0, "vp": 1.0, "head": 1.0, "chief": 1.0,
    })
    stretch_readiness: StretchReadinessConfig = field(default_factory=StretchReadinessConfig)


@dataclass
class StabilityConfig:
    strong_tenure_months: int = 36
    hopping_penalty: float = 0.30
    consecutive_short_tenure_threshold_months: int = 12
    consecutive_short_tenure_count: int = 3
    edu_bonus: dict = field(default_factory=lambda: {
        "tier_1": 0.05, "tier_2": 0.03, "tier_3": 0.01, "tier_4": 0.00,
    })


@dataclass
class PlatformConfig:
    passive_open_to_work_score: float = 0.40
    max_applications_norm: int = 10
    max_search_appearances_norm: int = 200
    max_response_time_norm_hrs: int = 200
    github_activity_max: float = 96.9
    max_endorsements_norm: int = 100


@dataclass
class CertBonusConfig:
    max_bonus: float = 0.10
    recency_decay_per_year: float = 0.10
    recency_floor: float = 0.50


@dataclass
class ThresholdConfig:
    semantic_fallback_discount: float = 0.60
    domain_relevance_min_cosine: float = 0.60
    thin_profile_char_limit: int = 50
    thin_profile_description_min: int = 20
    thin_profile_semantic_cap: float = 0.55


@dataclass
class RankingConfig:
    tiebreaker_composite_tolerance: float = 0.001


@dataclass
class PathConfig:
    raw_data: str = "data/raw/candidates.jsonl"
    processed_dir: str = "data/processed"
    output_dir: str = "data/outputs"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[str] = "logs/pipeline.log"


@dataclass
class Settings:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    skill_strength: SkillStrengthConfig = field(default_factory=SkillStrengthConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    cert_bonus: CertBonusConfig = field(default_factory=CertBonusConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path = Path("config/settings.yaml")) -> "Settings":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        emb = EmbeddingConfig(**raw.get("embedding", {}))
        llm = LLMConfig(**raw.get("llm", {}))
        ret = RetrievalConfig(**raw.get("retrieval", {}))
        sw = ScoringWeights(**raw.get("scoring_weights", {}))
        sk = SkillStrengthConfig(**raw.get("skill_strength", {}))

        traj_raw = raw.get("trajectory", {})
        sr_raw = traj_raw.pop("stretch_readiness", {})
        traj = TrajectoryConfig(
            seniority_levels=traj_raw.get("seniority_levels", TrajectoryConfig().seniority_levels),
            stretch_readiness=StretchReadinessConfig(**sr_raw),
        )

        stab = StabilityConfig(**raw.get("stability", {}))
        plat = PlatformConfig(**raw.get("platform", {}))
        cert = CertBonusConfig(**raw.get("cert_bonus", {}))
        thresh = ThresholdConfig(**raw.get("thresholds", {}))
        rank = RankingConfig(**raw.get("ranking", {}))
        pth = PathConfig(**raw.get("paths", {}))
        log = LoggingConfig(**raw.get("logging", {}))

        return cls(
            embedding=emb, llm=llm, retrieval=ret, scoring_weights=sw,
            skill_strength=sk, trajectory=traj, stability=stab,
            platform=plat, cert_bonus=cert, thresholds=thresh,
            ranking=rank, paths=pth, logging=log,
        )

    def override(self, **kwargs) -> "Settings":
        import copy
        new = copy.deepcopy(self)
        for key, value in kwargs.items():
            parts = key.split(".")
            obj = new
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        return new
