""" """

import re
from copy import deepcopy
from abc import ABC, abstractmethod
from overrides import overrides
from typing import Text, Dict
from registrable import Registrable
from .instances import (
    Instance,
    AtomicClaim
)
from .common import path_index
import logging


logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("core_post_processor.log")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BaseTemplate(ABC, Registrable):
    def __init__(
        self,
        template: Text
    ):
        """ Initialize a Template object
        with a string that has placeholders {{...}}
        where the the {{}}
        """
        self._template = template
         
    def __call__(self, instance: Instance) -> Text:
        """ """
        return self._render(instance)
    
    @abstractmethod
    def _render(self, instance: Instance) -> Text:
        """ """
        raise NotImplementedError("This method must be implemented in the subclass.")
    

@BaseTemplate.register("claim-formatting-template")
class ClaimFormattingTemplate(BaseTemplate):
    def __init__(self, template: Text):
        super().__init__(template)
        self._match_dict = self._process_template(template)
        
    def _process_template(self, template: Text) -> Dict[Text, Text]:
        """ """
        
        # find all {{}} including the brackets
        matches = re.findall(r"{{.*?}}", template, re.DOTALL)
        match_dict = {}
        
        if not matches:
            return {}
        for midx, match in enumerate(matches):
            handle = "{{@@" + f"{midx}" + "@@}}"
            template.replace(match, handle)
            match_dict[handle] = match
            
        return match_dict
    
    @overrides
    def _render(self, instance: Instance) -> Text:
        """ """
        assert isinstance(instance, AtomicClaim), "instance must be an AtomicClaim."
        template = self._template
        
        for handle, match in self._match_dict.items():
            content = path_index(instance, match)
            template = template.replace(handle, content)
            
        return template