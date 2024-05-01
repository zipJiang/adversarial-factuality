""" Common utilities that can be applied at multiple places in the project. """


from typing import List, Any, Callable, TypeVar, Dict, Text, Union


T = TypeVar("T")


def paginate_func(
    items: List[Any],
    page_size: int,
    func: Callable[..., T],
    combination: Callable[[List[T]], T]
) -> T:
    
    results = []
    for i in range(0, len(items), page_size):
        results.append(
            func(
                items[i:i+page_size]
            )
        )
        
    return combination(results)