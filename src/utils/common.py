""" Common utilities that can be applied at multiple places in the project. """

import logging
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from typing import List, Any, Callable, TypeVar, Dict, Text, Union, Iterable, Optional, Tuple
from langchain_interface.instances import Instance
from tqdm import tqdm


T = TypeVar("T")


def batched(
    items: List[T],
    batch_size: int
) -> List[List[T]]:
    """
    """
    
    return [
        items[i:i+batch_size]
        for i in range(0, len(items), batch_size)
    ]


def paginate_func(
    items: List[Any],
    page_size: int,
    func: Callable[..., T],
    combination: Optional[Callable[[List[T]], T]] = None,
    desc: Text = "Paginating",
    silent: bool = False
) -> Union[List[T], T]:
    
    results = []
    
    iterator = range(0, len(items), page_size)
    if not silent:
        iterator = tqdm(iterator, desc=desc)
        
    for i in iterator:
        results.append(
            func(
                items[i:i+page_size]
            )
        )
        
    return combination(results) if combination is not None else results


def stream_paginate_func(
    items: Iterable[Any],
    page_size: int,
    func: Callable[..., T],
    desc: Text = "Streaming",
    silent: bool = False,
    ignore_last: bool = False
) -> Iterable[T]:
    """This is a streaming function so that
    we cannot call len() on the items.
    """

    to_yield_input = []

    for item in tqdm(items, desc=desc, disable=silent):
        to_yield_input.append(item)
        if len(to_yield_input) >= page_size:
            yield func(to_yield_input)
            to_yield_input = []
            
    # if there are still items left
    if (not ignore_last) and to_yield_input:
        yield func(to_yield_input)
        
        
def path_index(
    item: Union[Dict, Instance],
    path: Union[Text, List[Text]],
    default: Optional[Any] = None,
    logger: Optional[logging.Logger] = None
):
    """ index a structured item with layer of indexing separated be :: """
    
    def _index(
        current_item: Union[Dict, Instance],
        separated_path: List[Text]
    ):
        if not separated_path:
            return current_item
        try:
            return _index(
                getattr(current_item, separated_path[0]) if not isinstance(current_item, dict) else current_item[separated_path[0]],
                separated_path[1:]
            )
        except (KeyError, AttributeError):
            if logger is not None:
                logger.warning(f"Failed to index {separated_path[0]} in {current_item} from {item}")
            return default
        
    return _index(
        item,
        path.split("::") if isinstance(path, str) else path
    )
    
    
def solve_milp(
    pairwise_entailment: np.ndarray,
    weighting: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """ """

    if pairwise_entailment.size == 0:
        return np.array([], np.int16), 0.0

    or_mat = np.tril(
        np.bitwise_or(pairwise_entailment, np.transpose(pairwise_entailment)),
    )

    indices = np.nonzero(or_mat)

    # TODO: add logging information

    constraints = np.zeros((len(indices[0]), len(weighting)), dtype=np.float32)

    constraints[np.arange(len(indices[0])), indices[0]] = 1
    constraints[np.arange(len(indices[1])), indices[1]] = 1

    res = milp(
        c=-weighting,
        integrality=np.ones_like(weighting),
        bounds=Bounds(
            lb=np.zeros_like(weighting) - 1e-8, ub=np.ones_like(weighting) + 1e-8
        ),
        constraints=(
            LinearConstraint(
                A=constraints,
                ub=np.ones(len(indices[0])) + 1e-8,
            ),
        ),
    )

    selection = res.x
    result = res.fun

    return selection, result



__MAX_LINE_PER_FILE__ = 10_000