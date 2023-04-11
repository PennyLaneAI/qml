#!/usr/bin/env python3

"""
This python file implements a very simple version of the "Least Connection" load balancing algorithm.
Read More on the least connection algorithm here:
  - https://www.cloudflare.com/en-gb/learning/performance/types-of-load-balancing-algorithms/
  - https://nginx.org/en/docs/http/ngx_http_upstream_module.html#least_conn
  - https://www.educative.io/answers/what-is-the-least-connections-load-balancing-technique

The `SortedWorkerHandler` implements this algorithm by assigning tasks to the Worker that currently
has the least load on it. The specific implementation of the algorithm in this module is specific to QML
and works most efficiently when the load of all tasks are known ahead of time, since that can be distributed the best.

Sample Usage:

>>> my_demo_loads = [
...    QMLDemo("QML_Demo_A", 7),
...    QMLDemo("QML_Demo_B", 4),
...    QMLDemo("QML_Demo_C", 1),
...    QMLDemo("QML_Demo_D", 9),
...    QMLDemo("QML_Demo_E", 16),
...    QMLDemo("QML_Demo_F", 23),
...    QMLDemo("QML_Demo_G", 1),
...    QMLDemo("QML_Demo_H", 4),
...]
>>> worker_handler = SortedWorkerHandler(num_workers=3)

The main thing to note here are the loads that will be distributed over the 3 workers:
 -> 7, 4, 1, 9, 16, 23, 1, 4

>>> worker_handler.add_task(*my_demo_loads)
>>> worker_handler.assign_tasks_to_workers()

The first thing that will happen is that the tasks will be sorted from the highest load to lowest:
  -> 23, 16, 9, 7, 4, 4, 1, 1

Next the first task will be assigned to the worker that currently has the lowest load. The SortedWorkerHandler maintains
a sorted list of workers internally, after a task is added to a worker, the list is re-sorted such that
the first element is always the worker currently with the least amount of load.

So let's say we have the following workers: [0, 1, 2]

The tasks will be distributed as following:

Task Load  |  Worker ID  |  Total Load on Worker after this task is added
--------------------------------------------------------------------------
  23       |     0       |    23
  16       |     1       |    16
  9        |     2       |     9
  7        |     2       |    16
  4        |     1       |    20
  4        |     2       |    20
  1        |     1       |    21
  1        |     1       |    21
Note: The `Worker ID` here is added for illustrative purposes, there is no worker id needed in the actual implementation
      as the list indexes do the job much better.

So at the end the total load on each worker is: 23, 21, 21
Which is the closest equal distribution of that list of tasks possible.

The `load` here refers to the amount of time it takes to execute a sphinx demo.

Once all the tasks have been passed to the worker handler:

>>> worker_handler.workers

Will return the list of workers with the tasks distributed

Another example:

>>> my_demo_loads = [
...    QMLDemo("QML_Demo_A", 1),
...    QMLDemo("QML_Demo_B", 1),
...    QMLDemo("QML_Demo_C", 1),
...    QMLDemo("QML_Demo_D", 6),
...    QMLDemo("QML_Demo_E", 12),
...    QMLDemo("QML_Demo_F", 9)
...]
>>> worker_handler = SortedWorkerHandler(num_workers=2)
>>> worker_handler.add_task(*my_demo_loads)
>>> worker_handler.assign_tasks_to_workers()

Step 1: sort all loads from the highest to lowest:
  -> 12, 9, 6, 1, 1, 1

Step 2: Create two workers Worker(0) and Worker(1)


Step 3: Distribute:
Task Load  |  Worker ID  |  Total Load on Worker after this task is added
--------------------------------------------------------------------------
  12       |     0       |    12
   9       |     1       |     9
   6       |     1       |    15
   1       |     0       |    13
   1       |     0       |    14
   1       |     0       |    15

Final workload:
  - Worker(0) = 15
  - Worker(1) = 15

Serializing the result:

>>> json.dumps(worker_handler, cls=WorkerAndTaskJSONEncoder)
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Union


@dataclass(frozen=True)
class ReturnTypes:
    DictQMLDemo = Dict[str, Union[int, float, str, Dict[str, Any]]]
    DictWorker = Dict[str, Union[int, float, DictQMLDemo]]
    DictSortedWorkerHandler = Dict[str, Union[int, List[DictWorker]]]


@dataclass
class QMLDemo:
    """
    A simple struct representing a QML Demo. The most important attribute here is the `load`, however
    the name is required to distinguish the Demo once it has been assigned to a worker.

    Args:
        name (str): The name of the Demo, can be path to demo as string.
        load (int, float): The amount of time it takes sphinx to execute this demo.
        metadata (Dict[str, Any]): Any additional (and optional) information about the demo that can be stored at
                                   initialization for usage later.
    """

    name: str
    load: Union[int, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class Worker:
    """
    Represent a worker and is used to track all the tasks assigned to a specific instance of Worker.
    """

    def __init__(self):
        self.__tasks: List[QMLDemo] = []

    @property
    def load(self) -> Union[int, float]:
        return sum([task.load for task in self.__tasks])

    @property
    def tasks(self) -> List[QMLDemo]:
        return sorted(self.__tasks, key=lambda t: t.name)

    # this add_task should not be used directly. See add_task in `SortedWorkerHandler` instead
    def add_task(self, task: QMLDemo) -> None:
        self.__tasks.append(task)

    def __repr__(self):
        return f"<Worker({self.tasks})>"

    def asdict(self) -> ReturnTypes.DictWorker:
        return {"load": self.load, "tasks": [asdict(t) for t in self.tasks]}


class SortedWorkerHandler:
    """
    Manages a sorted list of `Worker` class instances.

    Tasks should be added to this class instead of an instance of a Worker class directly. This class will manage and
    track the loads of all the workers.

    The class buffers all incoming tasks in an internal list and distributes the load across workers once `assign_tasks_to_workers`
    is called.
    """

    def __init__(self, num_workers: int):
        self.__task_buffer: List[QMLDemo] = []
        self.__workers: List[Worker] = [Worker() for _ in range(num_workers)]

    @property
    def workers(self) -> List[Worker]:
        return self.__workers

    @property
    def num_workers(self) -> int:
        return len(self.__workers)

    def add_task(self, *task: QMLDemo) -> None:
        self.__task_buffer.extend(task)

    def __assign_task_to_workers(self, task: QMLDemo) -> None:
        min_loaded_worker = self.__workers.pop(0)
        min_loaded_worker.add_task(task)

        if self.num_workers == 0:  # There is only 1 worker
            self.__workers.append(min_loaded_worker)
            return None

        new_load_on_worker = min_loaded_worker.load

        max_loaded_worker = self.__workers[-1].load
        index_to_insert_worker = self.num_workers
        if new_load_on_worker < max_loaded_worker:
            for i, worker in enumerate(self.__workers):
                if worker.load > new_load_on_worker:
                    index_to_insert_worker = i
                    break
        self.__workers.insert(index_to_insert_worker, min_loaded_worker)

        return None

    def assign_tasks_to_workers(self) -> None:
        sorted_tasks = sorted(self.__task_buffer, key=lambda t: t.load, reverse=True)
        for task in sorted_tasks:
            self.__assign_task_to_workers(task)
        self.__task_buffer: List[QMLDemo] = []

    def __getitem__(self, index: int) -> Worker:
        return self.__workers[index]

    def asdict(self) -> ReturnTypes.DictSortedWorkerHandler:
        return {"num_workers": self.num_workers, "workers": [wk.asdict() for wk in self.workers]}


class WorkerAndTaskJSONEncoder(json.JSONEncoder):
    def default(self, o: Union[QMLDemo, Worker, SortedWorkerHandler]):
        if isinstance(o, QMLDemo):
            return asdict(o)
        elif isinstance(o, Worker):
            return {"load": o.load, "tasks": [self.default(t) for t in o.tasks]}
        elif isinstance(o, SortedWorkerHandler):
            return {"num_workers": o.num_workers, "workers": [self.default(wk) for wk in o.workers]}
        else:
            return super().default(o)
