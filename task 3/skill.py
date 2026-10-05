
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, FrozenSet, List, Optional
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Configuration / domain objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shift:
    """A recurring availability window: which weekdays, and what daily hours."""

    timezone: str
    business_days: FrozenSet[int]  # Monday=0 ... Sunday=6
    start_time: time
    end_time: time

    def __post_init__(self):
        if not self.timezone:
            raise ValueError("Shift.timezone is required.")
        try:
            ZoneInfo(self.timezone)
        except Exception as exc:
            raise ValueError(f"Invalid timezone '{self.timezone}': {exc}") from exc
        if not self.business_days:
            raise ValueError("Shift.business_days must not be empty.")
        if not self.business_days.issubset(set(range(7))):
            raise ValueError("Shift.business_days must contain values 0-6 (Mon-Sun).")
        if self.start_time >= self.end_time:
            raise ValueError("Shift.start_time must be before end_time.")

    def is_within(self, moment: datetime) -> bool:
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("Shift.is_within requires a timezone-aware datetime.")
        local = moment.astimezone(ZoneInfo(self.timezone))
        return local.weekday() in self.business_days and self.start_time <= local.time() < self.end_time


@dataclass(frozen=True)
class Agent:
    """A support agent's routing-relevant attributes."""

    agent_id: str
    skills: FrozenSet[str]
    languages: FrozenSet[str]
    technical_tier: int  # highest tier this agent can independently handle
    capacity: int  # max concurrent tickets
    shift: Shift
    current_workload: int = 0
    is_available: bool = True  # real-time status (e.g. online / on break)

    def __post_init__(self):
        if self.capacity < 0:
            raise ValueError(f"Agent '{self.agent_id}': capacity must be >= 0.")
        if self.current_workload < 0:
            raise ValueError(f"Agent '{self.agent_id}': current_workload must be >= 0.")
        if self.technical_tier < 0:
            raise ValueError(f"Agent '{self.agent_id}': technical_tier must be >= 0.")

    def has_capacity(self) -> bool:
        return self.current_workload < self.capacity


@dataclass(frozen=True)
class Ticket:
    """A ticket awaiting routing."""

    ticket_id: str
    required_skill: str
    language: str
    required_tier: int
    priority: str
    created_at: datetime

    def __post_init__(self):
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("Ticket.created_at must be a timezone-aware datetime.")


@dataclass(frozen=True)
class RoutingConfig:
    """Configurable routing rules."""

    operating_hours: Shift  # global "business is open" window, for after-hours logic
    priority_rank: Dict[str, int]  # lower rank = higher priority; must be exhaustive
    emergency_priority: str  # which priority value triggers the emergency queue
    no_capacity_policy: str = "backlog"  # "backlog" or "overflow"
    known_skills: Optional[FrozenSet[str]] = None  # if set, unknown skills are rejected safely
    known_languages: Optional[FrozenSet[str]] = None

    def __post_init__(self):
        if not self.priority_rank:
            raise ValueError("RoutingConfig.priority_rank must not be empty.")
        if self.emergency_priority not in self.priority_rank:
            raise ValueError(
                f"RoutingConfig.emergency_priority '{self.emergency_priority}' "
                f"is not in priority_rank {sorted(self.priority_rank)}."
            )
        if self.no_capacity_policy not in ("backlog", "overflow"):
            raise ValueError("RoutingConfig.no_capacity_policy must be 'backlog' or 'overflow'.")

    def rank_of(self, priority: str) -> int:
        if priority not in self.priority_rank:
            raise KeyError(
                f"Unknown priority '{priority}'. Configured priorities: "
                f"{sorted(self.priority_rank)}."
            )
        return self.priority_rank[priority]


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------

class RoutingOutcome(str, Enum):
    ASSIGNED = "assigned"
    EMERGENCY_QUEUE = "emergency_queue"
    SCHEDULED_QUEUE = "scheduled_queue"
    BACKLOG = "backlog"
    OVERFLOW = "overflow"


@dataclass(frozen=True)
class AgentEvaluation:
    """Transparency record of why an agent was or wasn't chosen."""

    agent_id: str
    eligible: bool
    reason: str


@dataclass(frozen=True)
class RoutingDecision:
    ticket_id: str
    outcome: RoutingOutcome
    reason: str
    agent_id: Optional[str] = None
    queue: Optional[str] = None
    evaluated_agents: tuple = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Core routing logic (pure function)
# ---------------------------------------------------------------------------

def _skill_language_tier_eligible(agent: Agent, ticket: Ticket) -> bool:
    return (
        ticket.required_skill in agent.skills
        and ticket.language in agent.languages
        and agent.technical_tier >= ticket.required_tier
    )


def route_ticket(ticket: Ticket, agents: List[Agent], config: RoutingConfig) -> RoutingDecision:
    """
    Deterministically decide where a ticket should go. Pure function: does
    not mutate `agents`. Same (ticket, agents, config) always yields the
    same `RoutingDecision`.
    """
    # Safety / config checks: never invent capabilities for unknown skills
    # or languages -- fail safely to overflow instead of guessing a match.
    if config.known_skills is not None and ticket.required_skill not in config.known_skills:
        return RoutingDecision(
            ticket_id=ticket.ticket_id,
            outcome=RoutingOutcome.OVERFLOW,
            reason=f"Unrecognized skill '{ticket.required_skill}'; cannot safely match an agent.",
            queue="overflow",
        )
    if config.known_languages is not None and ticket.language not in config.known_languages:
        return RoutingDecision(
            ticket_id=ticket.ticket_id,
            outcome=RoutingOutcome.OVERFLOW,
            reason=f"Unrecognized language '{ticket.language}'; cannot safely match an agent.",
            queue="overflow",
        )
    # Validate priority is known (raises KeyError deliberately -- do not guess a rank).
    config.rank_of(ticket.priority)

    # Evaluate every agent, deterministically ordered by agent_id.
    ordered_agents = sorted(agents, key=lambda a: a.agent_id)
    evaluations: List[AgentEvaluation] = []
    capacity_eligible: List[Agent] = []

    for agent in ordered_agents:
        if not _skill_language_tier_eligible(agent, ticket):
            evaluations.append(AgentEvaluation(agent.agent_id, False, "skill/language/tier mismatch"))
            continue
        if not agent.shift.is_within(ticket.created_at):
            evaluations.append(AgentEvaluation(agent.agent_id, False, "off-shift"))
            continue
        if not agent.is_available:
            evaluations.append(AgentEvaluation(agent.agent_id, False, "unavailable"))
            continue
        if not agent.has_capacity():
            evaluations.append(AgentEvaluation(agent.agent_id, False, "over capacity"))
            continue
        evaluations.append(AgentEvaluation(agent.agent_id, True, "eligible and available"))
        capacity_eligible.append(agent)

    if capacity_eligible:
        # Deterministic tie-break: least loaded agent first, then agent_id.
        best = min(capacity_eligible, key=lambda a: (a.current_workload, a.agent_id))
        return RoutingDecision(
            ticket_id=ticket.ticket_id,
            outcome=RoutingOutcome.ASSIGNED,
            reason=(
                f"Best eligible agent by lowest current workload "
                f"({best.current_workload}/{best.capacity})."
            ),
            agent_id=best.agent_id,
            evaluated_agents=tuple(evaluations),
        )

    # No agent currently eligible+available+with-capacity.
    any_skill_language_tier_match = any(_skill_language_tier_eligible(a, ticket) for a in agents)
    if not any_skill_language_tier_match:
        return RoutingDecision(
            ticket_id=ticket.ticket_id,
            outcome=RoutingOutcome.OVERFLOW,
            reason=(
                f"No configured agent has skill='{ticket.required_skill}', "
                f"language='{ticket.language}', tier>={ticket.required_tier}."
            ),
            queue="overflow",
            evaluated_agents=tuple(evaluations),
        )

    after_hours = not config.operating_hours.is_within(ticket.created_at)
    if after_hours:
        if ticket.priority == config.emergency_priority:
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                outcome=RoutingOutcome.EMERGENCY_QUEUE,
                reason=(
                    f"After operating hours and priority is '{ticket.priority}' "
                    "(emergency priority) -> on-call emergency queue."
                ),
                queue="emergency",
                evaluated_agents=tuple(evaluations),
            )
        return RoutingDecision(
            ticket_id=ticket.ticket_id,
            outcome=RoutingOutcome.SCHEDULED_QUEUE,
            reason="After operating hours and priority is not emergency-level -> scheduled queue.",
            queue="scheduled",
            evaluated_agents=tuple(evaluations),
        )

    # Business hours, matching agents exist, but none are available/have capacity.
    if config.no_capacity_policy == "backlog":
        return RoutingDecision(
            ticket_id=ticket.ticket_id,
            outcome=RoutingOutcome.BACKLOG,
            reason="Matching agents exist but none have capacity right now -> backlog buffer.",
            queue="backlog",
            evaluated_agents=tuple(evaluations),
        )
    return RoutingDecision(
        ticket_id=ticket.ticket_id,
        outcome=RoutingOutcome.OVERFLOW,
        reason="Matching agents exist but none have capacity right now -> overflow pool.",
        queue="overflow",
        evaluated_agents=tuple(evaluations),
    )


# ---------------------------------------------------------------------------
# Stateful wrapper: commits side effects (workload bump / queue insertion)
# ---------------------------------------------------------------------------

@dataclass
class QueuedTicket:
    ticket_id: str
    priority_rank: int
    created_at: datetime
    sequence: int  # insertion order, for FIFO tie-break


class Router:
    """
    Stateful convenience wrapper around `route_ticket`. Holds the live
    agent roster and the backlog/overflow/emergency/scheduled queues, and
    commits the effects of a routing decision (workload increment, or
    queue insertion).
    """

    def __init__(self, agents: Dict[str, Agent], config: RoutingConfig):
        self.agents = agents  # agent_id -> Agent
        self.config = config
        self.backlog: List[QueuedTicket] = []
        self.overflow: List[QueuedTicket] = []
        self.emergency_queue: List[QueuedTicket] = []
        self.scheduled_queue: List[QueuedTicket] = []
        self._sequence = 0

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _enqueue(self, target: List[QueuedTicket], ticket: Ticket) -> None:
        target.append(
            QueuedTicket(
                ticket_id=ticket.ticket_id,
                priority_rank=self.config.rank_of(ticket.priority),
                created_at=ticket.created_at,
                sequence=self._next_sequence(),
            )
        )
        # Keep queues ordered by priority first, then FIFO (created_at, sequence).
        target.sort(key=lambda q: (q.priority_rank, q.created_at, q.sequence))

    def route(self, ticket: Ticket) -> RoutingDecision:
        decision = route_ticket(ticket, list(self.agents.values()), self.config)

        if decision.outcome == RoutingOutcome.ASSIGNED:
            agent = self.agents[decision.agent_id]
            self.agents[decision.agent_id] = Agent(
                agent_id=agent.agent_id,
                skills=agent.skills,
                languages=agent.languages,
                technical_tier=agent.technical_tier,
                capacity=agent.capacity,
                shift=agent.shift,
                current_workload=agent.current_workload + 1,
                is_available=agent.is_available,
            )
        elif decision.outcome == RoutingOutcome.BACKLOG:
            self._enqueue(self.backlog, ticket)
        elif decision.outcome == RoutingOutcome.OVERFLOW:
            self._enqueue(self.overflow, ticket)
        elif decision.outcome == RoutingOutcome.EMERGENCY_QUEUE:
            self._enqueue(self.emergency_queue, ticket)
        elif decision.outcome == RoutingOutcome.SCHEDULED_QUEUE:
            self._enqueue(self.scheduled_queue, ticket)

        return decision


if __name__ == "__main__":
    business_hours = Shift(
        timezone="UTC",
        business_days=frozenset({0, 1, 2, 3, 4}),
        start_time=time(9, 0),
        end_time=time(17, 0),
    )
    agent = Agent(
        agent_id="A1",
        skills=frozenset({"billing"}),
        languages=frozenset({"en"}),
        technical_tier=2,
        capacity=3,
        shift=business_hours,
    )
    config = RoutingConfig(
        operating_hours=business_hours,
        priority_rank={"P1": 0, "P2": 1, "P3": 2},
        emergency_priority="P1",
    )
    ticket = Ticket(
        ticket_id="T-1",
        required_skill="billing",
        language="en",
        required_tier=1,
        priority="P2",
        created_at=datetime(2024, 5, 1, 10, 0, tzinfo=ZoneInfo("UTC")),
    )
    router = Router({agent.agent_id: agent}, config)
    print(router.route(ticket))