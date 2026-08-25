from .event_loop import SimulationEventLoop

__all__ = ["EdgeSpecSimulator", "SimulationEventLoop"]


def __getattr__(name: str):
    if name == "EdgeSpecSimulator":
        from .simulator import EdgeSpecSimulator

        return EdgeSpecSimulator
    raise AttributeError(name)
