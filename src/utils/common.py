""" Common utilities that can be applied at multiple places in the project. """


from typing import List, Any, Callable, TypeVar, Dict, Text, Union
from tqdm import tqdm


T = TypeVar("T")


def paginate_func(
    items: List[Any],
    page_size: int,
    func: Callable[..., T],
    combination: Callable[[List[T]], T],
    silent: bool = False
) -> T:
    
    results = []
    
    iterator = range(0, len(items), page_size)
    if not silent:
        iterator = tqdm(iterator, desc="Paginating")
        
    for i in iterator:
        results.append(
            func(
                items[i:i+page_size]
            )
        )
        
    return combination(results)