from typing import Callable, TYPE_CHECKING
import simpy

if TYPE_CHECKING:
    from train_lib.models import Train


class Simulation(simpy.Environment):
    def __init__(self):
        super().__init__()
        self.trains: list["Train"] = []

    def add_train(self, train: "Train"):
        self.trains.append(train)
        self.process(train.run())

    # def setup_network(self, network: Callable):
    #     network()
    #     return
