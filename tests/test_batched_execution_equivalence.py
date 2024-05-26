"""Run batched execution and compare with single execution
to see whether there results correlate.
"""

import os
from envyaml import EnvYAML
from unittest import TestCase
from copy import deepcopy
from random import Random
from src.aggregator import (
    FActScoreAggregator
)
from src.abstention_detector import FActScoreAbstentionDetector
from src.retriever.retriever import Retriever
from src.utils.instances import ScorerInstance
from src.decomposer.factscore_decomposer import FActScoreDecomposer
from src.decomposer.deduplicated_decompser import DeduplicatedDecomposer
from src.scorer import (
    Scorer,
    DecomposeScorer,
    LLMGeneralCheckWorthyScorer,
    LLMSpecificCheckWorthyScorer,
    UNLIConfidenceBoostScorer,
    LLMSupportScorer
)
from src.entailer import (
    SoftEntailer,
    Entailer
)


class TestBatchedExecutionEquivalence(TestCase):
    """
    """
    def setUp(self):
        """
        """
        from langchain.globals import set_llm_cache
        from langchain.cache import SQLiteCache
        set_llm_cache(SQLiteCache(".cache/.test_batchified_cache.db"))
        # we don't do cache, so this test LLM generation each time.
        self.test_cases = [
            ScorerInstance(
                text="Kalki Koechlin is an Indian actress and writer known for her work in Hindi films. Born on January 10, 1984, in Pondicherry, India, Kalki spent her early years in Auroville before moving to London to study drama and theatre. She made her acting debut in the critically acclaimed film \"Dev.D\" in 2009, for which she received the Filmfare Award for Best Supporting Actress.\n\nKalki has since appeared in a variety of films, showcasing her versatility as an actress. Some of her notable performances include \"Zindagi Na Milegi Dobara,\" \"Margarita with a Straw,\" and \"Gully Boy.\" She has garnered praise for her unconventional choice of roles and her ability to portray complex characters with depth and authenticity.\n\nIn addition to her acting career, Kalki is also a talented writer and has written for various publications on topics such as feminism, mental health, and social issues. She is known for her outspoken views on gender equality and has been a vocal advocate for women's rights in India. Kalki Koechlin continues to be a prominent figure in the Indian film industry, known for her bold and fearless approach to her craft.",
                topic="Kalki Koechlin"),
            ScorerInstance(
                text="Adil Rami is a professional French footballer who was born on December 27, 1985, in Bastia, France. He primarily plays as a central defender and is known for his physicality, aerial prowess, and strong defensive abilities. Rami began his professional career at Lille OSC in 2006 and quickly established himself as a key player in the team's defense.\n\nIn 2011, Rami made a move to Spanish club Valencia CF, where he continued to impress with his performances. He later joined Italian giants AC Milan in 2014 before returning to France to play for Marseille in 2017. Rami has also represented the French national team, earning caps and playing a key role in their success at the 2018 FIFA World Cup, where they emerged as champions.\n\nOff the field, Rami is known for his charismatic personality and has gained a significant following on social media. He is also involved in various charitable activities and has shown a commitment to giving back to the community. Rami's career has been marked by success at both the club and international levels, making him one of the prominent figures in French football.",
                topic="Adil Rami"),
            ScorerInstance(
                text="Song Kang is a South Korean actor born on April 23, 1994. He gained popularity for his roles in various Korean dramas such as \"Love Alarm,\" \"Navillera,\" and \"Nevertheless.\" Song Kang began his acting career in 2017 and quickly rose to fame for his charming looks and versatile acting skills. He is known for his ability to portray a wide range of characters, from sweet and romantic to intense and brooding. With his rising popularity both in South Korea and internationally, Song Kang has become one of the most sought-after young actors in the Korean entertainment industry.",
                topic="Song Kang"),
            ScorerInstance(
                text=" Rakhi Sawant is a renowned Indian actress, model, dancer, and television personality. She was born on November 27, 1978, in Mumbai, Maharashtra, India. Rakhi began her career in the entertainment industry as a contestant on the reality show \"Bigg Boss\" in 2006. Since then, she has appeared in numerous Bollywood films, reality shows, and television series.\n\nRakhi made her acting debut in the film \"Agasthya\" in 2004, and her first major role was in the film \"Masti\" in 2004. She has since appeared in several other films, including \"Shararat,\" \"Krishna Cottage,\" \"Main Hoon Na,\" and \"Bhootwala Serial Killer.\"\n\nApart from acting, Rakhi has also been a popular contestant on various reality shows, including \"Nach Baliye,\" \"Jhalak Dikhhla Jaa,\" and \"Fear Factor.\" She has also hosted several television shows, including \"Rakhi Ka Swayamvar,\" \"Rakhi Ka Insaaf,\" and \"Rakhi Ki Salaam-e-Ishq.\"\n\nRakhi is known for her bold and controversial personality, and she has been a subject of media attention for her public appearances and statements. She has also been involved in several controversies throughout her career, including her alleged relationship with actor Abhishek Aasman Kadam and her participation in a reality show where she claimed to be married to an American businessman.\n\nDespite the controversies, Rakhi continues to be a popular figure in the Indian entertainment industry. She has a large following on social media and is known for her outspoken nature and her ability to entertain her fans. She is also an active philanthropist and has worked with several NGOs to raise awareness about various social issues.",
                topic="Rakhi Sawant",
            ),
            ScorerInstance(
                text="Sian Massey-Ellis is a highly respected and accomplished sports official, best known for her work as a referee in the English Football League (EFL) and the Football Association (FA) in the United Kingdom. Born on March 2, 1983, in Liverpool, England, Massey-Ellis grew up with a strong passion for sports, particularly football.\n\nAfter completing her education, Massey-Ellis began her career in officiating in 2003, starting with local and grassroots football. She quickly progressed through the ranks, earning her place on the National List of Referees in 2008. In 2011, she became the first female referee to officiate a Football League match in England, taking charge of a League Two fixture between Torquay United and Morecambe.\n\nMassey-Ellis has since become a regular fixture in the EFL, officiating numerous matches at various levels. She has also made history by becoming the first female referee to officiate a Championship game in 2014, as well as the first female referee to officiate a League Cup match in 2015.\n\nIn addition to her work in the EFL, Massey-Ellis has also officiated in the FA Women's Super League and the FA Women's Championship. She has been a referee for the FA Cup and the FA WSL Continental Tyres Cup, as well as the FA Women's League Cup.\n\nMassey-Ellis has also officiated internationally, including matches in the UEFA Women's Champions League and the UEFA European Women's Under-19 Championship. She was also selected to officiate at the 2019 FIFA Women's World Cup in France.\n\nMassey-Ellis has been recognized for her achievements with numerous awards, including the FA Women's Football Award for Referee of the Year in 2014 and 2015, and the FA Women's Football Award for Services to Refereeing in 2019. She was also named the FA WSL Referee of the Year in 2018.\n\nOff the field, Massey-Ellis is an advocate for women in sport and works to promote equality and diversity in football. She is a mentor for the FA's NextGen programme, which aims to encourage more women to become referees, and she also works with schools and community groups to inspire the next generation of footballers and officials.",
                topic="Sian Massey-Ellis",
            )
        ]

        # We explicitly provide the cases here so that we can compare the results
        # at a finer granularity.
        self.test_decomposed_cases_raw = [
            [
                "Kalki Koechlin is an Indian actress.",
                "Kalki Koechlin is a writer.",
                "Kalki Koechlin is known for her work.",
                "Kalki Koechlin is known for her work in Hindi films.",
                "Kalki was born on January 10, 1984.",
                "Kalki was born in Pondicherry, India.",
                "Kalki spent her early years in Auroville.",
                "Kalki moved to London.",
                "Kalki moved to London to study.",
                "Kalki moved to London to study drama and theatre.",
                "She made her acting debut.",
                "She made her acting debut in a film.",
                "\"Dev\" is a film.",
                "\"Dev\" is a critically acclaimed film.",
                "D\" was released in 2009.",
                "She received an award.",
                "She received the Filmfare Award.",
                "She received the Filmfare Award for Best Supporting Actress.",
                "She received the Filmfare Award for Best Supporting Actress for D\".",
                "Kalki has appeared in a variety of films.",
                "Kalki has appeared in a variety of films since.",
                "Kalki has showcased her versatility as an actress.",
                "Kalki has showcased her versatility as an actress in films.",
                "Some of her performances include \"Zindagi Na Milegi Dobara.\"",
                "\"Zindagi Na Milegi Dobara\" is one of her notable performances.",
                "Some of her performances include \"Margarita with a Straw.\"",
                "\"Margarita with a Straw\" is one of her notable performances.",
                "Some of her performances include \"Gully Boy.\"",
                "\"Gully Boy\" is one of her notable performances.",
                "She has garnered praise.",
                "She has garnered praise for her acting.",
                "She has garnered praise for her acting choices.",
                "She has garnered praise for her unconventional choice of roles.",
                "She has the ability to portray complex characters.",
                "She has the ability to portray complex characters with depth.",
                "She has the ability to portray complex characters with authenticity.",
                "Kalki has an acting career.",
                "Kalki is a talented writer.",
                "Kalki has written for various publications.",
                "Kalki has written for various publications on topics such as feminism.",
                "Kalki has written for various publications on topics such as mental health.",
                "Kalki has written for various publications on topics such as social issues.",
                "She is known for her outspoken views.",
                "She is known for her outspoken views on gender equality.",
                "She has been a vocal advocate.",
                "She has been a vocal advocate for women's rights.",
                "She has been a vocal advocate for women's rights in India.",
                "Kalki Koechlin is a prominent figure."
                "Kalki Koechlin is a prominent figure in the Indian film industry.",
                "Kalki Koechlin is known for her bold approach.",
                "Kalki Koechlin is known for her fearless approach.",
                "Kalki Koechlin is known for her bold and fearless approach.",
                "Kalki Koechlin is known for her bold and fearless approach to her craft."
            ],
            [
                "Adil Rami is a professional French footballer.",
                "Adil Rami was born on December 27, 1985.",
                "Adil Rami was born in Bastia, France.",
                "He primarily plays as a central defender.",
                "He is known for his physicality.",
                "He is known for his aerial prowess.",
                "He is known for his strong defensive abilities.",
                "Rami began his professional career.",
                "Rami began his professional career at Lille OSC.",
                "Rami began his professional career at Lille OSC in 2006.",
                "Rami quickly established himself.",
                "Rami quickly established himself as a key player.",
                "Rami quickly established himself as a key player in the team's defense.",
                "Rami quickly established himself as a key player in the team's defense at Lille OSC.",
                "Rami quickly established himself as a key player in the team's defense at Lille OSC in 2006.",
                "Rami made a move to Valencia CF in 2011.",
                "Rami continued to impress with his performances.",
                "Rami made a move to Valencia CF in 2011 and continued to impress with his performances.",
                "In 2011, Rami joined Valencia CF.",
                "In 2011, Rami's performances were impressive at Valencia CF.",
                "He later joined AC Milan.",
                "He joined AC Milan in 2014.",
                "He later returned to France.",
                "He played for Marseille in 2017.",
                "Rami has represented the French national team.",
                "Rami earned caps.",
                "Rami played a key role.",
                "Rami's team was successful.",
                "Rami's team was successful at the 2018 FIFA World Cup.",
                "The 2018 FIFA World Cup was held.",
                "France emerged as champions.",
                "Rami has a charismatic personality.",
                "Rami is known for his charismatic personality.",
                "Rami has gained a following on social media.",
                "Rami has gained a significant following on social media.",
                "Social media is a platform for Rami's following.",
                "Rami has a following on social media.",
                "He is involved in charitable activities.",
                "He has shown a commitment to giving back to the community.",
                "Rami has been successful at the club level.",
                "Rami has been successful at the international level.",
                "Rami is one of the prominent figures.",
                "Rami is one of the prominent figures in French football."
            ],
            [
                "Song Kang is a South Korean actor.",
                "Song Kang was born on April 23, 1994.",
                "He gained popularity.",
                "He gained popularity for his roles.",
                "He gained popularity for his roles in Korean dramas.",
                "\"Love Alarm\" is a Korean drama.",
                "He appeared in \"Love Alarm.\"",
                "\"Navillera\" is a Korean drama.",
                "He appeared in \"Navillera.\"",
                "\"Nevertheless\" is a Korean drama.",
                "He appeared in \"Nevertheless.\"",
                "Song Kang began his acting career in 2017.",
                "Song Kang rose to fame.",
                "Song Kang rose to fame for his charming looks.",
                "Song Kang rose to fame for his versatile acting skills.",
                "He is known."
                "He is known for his ability."
                "He is known for his ability to portray.",
                "He is known for his ability to portray a wide range.",
                "He is known for his ability to portray a wide range of characters.",
                "He can portray sweet and romantic characters.",
                "He can portray intense and brooding characters."
                "Song Kang is a young actor.",
                "Song Kang is popular in South Korea.",
                "Song Kang is popular internationally.",
                "Song Kang is sought-after.",
                "Song Kang is one of the most sought-after young actors.",
                "Song Kang is one of the most sought-after young actors in the Korean entertainment industry.",
            ],
            [
                "Rakhi Sawant is an Indian actress.",
                "Rakhi Sawant is a renowned Indian actress.",
                "Rakhi Sawant is a model.",
                "Rakhi Sawant is a dancer.",
                "Rakhi Sawant is a television personality.",
                "She was born.",
                "She was born on November 27, 1978.",
                "She was born in Mumbai.",
                "She was born in Mumbai, Maharashtra.",
                "She was born in India.",
                "Rakhi began her career.",
                "Rakhi began her career in the entertainment industry.",
                "Rakhi began her career as a contestant.",
                "Rakhi began her career as a contestant on a reality show.",
                "\"Bigg Boss\" is a reality show.",
                "Rakhi began her career as a contestant on \"Bigg Boss\".",
                "Rakhi began her career as a contestant on \"Bigg Boss\" in 2006.",
                "She has appeared in numerous Bollywood films.",
                "She has appeared in numerous Bollywood films since then.",
                "She has appeared in reality shows.",
                "She has appeared in reality shows since then.",
                "She has appeared in television series.",
                "She has appeared in television series since then.",
                "Rakhi made her acting debut.",
                "Rakhi made her acting debut in the film \"Agasthya\".",
                "\"Agasthya\" is a film.",
                "Rakhi made her acting debut in \"Agasthya\" in 2004.",
                "Rakhi had her first major role.",
                "Rakhi had her first major role in the film \"Masti\".",
                "\"Masti\" is a film.",
                "Rakhi had her first major role in \"Masti\" in 2004.",
                "She has appeared in several other films.",
                "\"Shararat\" is a film."
                "She has appeared in \"Shararat.\"",
                "\"Krishna Cottage\" is a film.",
                "She has appeared in \"Krishna Cottage.\"",
                "\"Main Hoon Na\" is a film.",
                "She has appeared in \"Main Hoon Na.\"",
                "\"Bhootwala Serial Killer\" is a film.",
                "She has appeared in \"Bhootwala Serial Killer.\"",
                "Rakhi has acted.",
                "Rakhi has acted in various reality shows.",
                "\"Nach Baliye\" is a reality show.",
                "Rakhi has been a contestant on \"Nach Baliye.\"",
                "\"Jhalak Dikhhla Jaa\" is a reality show.",
                "Rakhi has been a contestant on \"Jhalak Dikhhla Jaa.\"",
                "\"Fear Factor\" is a reality show.",
                "Rakhi has been a contestant on \"Fear Factor.\"",
                "She has hosted several television shows.",
                "Rakhi Ka Swayamvar is a television show.",
                "She has hosted Rakhi Ka Swayamvar.",
                "Rakhi Ka Insaaf is a television show.",
                "She has hosted Rakhi Ka Insaaf.",
                "Rakhi Ki Salaam-e-Ishq is a television show.",
                "She has hosted Rakhi Ki Salaam-e-Ishq.",
                "Rakhi is known for her bold personality.",
                "Rakhi is known for her controversial personality.",
                "She has been a subject of media attention.",
                "She has been a subject of media attention for her public appearances.",
                "She has been a subject of media attention for her statements.",
                "She has been involved in several controversies.",
                "She has been involved in several controversies throughout her career.",
                "She was allegedly involved in a relationship with Abhishek Aasman Kadam.",
                "She participated in a reality show.",
                "She claimed to be married to an American businessman on the reality show.",
                "Rakhi is a popular figure.",
                "Rakhi is a popular figure in the Indian entertainment industry.",
                "Despite controversies.",
                "Despite controversies, Rakhi continues to be a popular figure.",
                "She has a large following on social media.",
                "She is known for her outspoken nature.",
                "She is known for her ability to entertain her fans.",
                "She is an active philanthropist.",
                "She has worked with several NGOs.",
                "She has worked with several NGOs to raise awareness.",
                "She has worked with several NGOs to raise awareness about various social issues.",
            ],
            [
                "Sian Massey-Ellis is a highly respected and accomplished sports official.",
                "Sian Massey-Ellis is best known for her work as a referee.",
                "Sian Massey-Ellis works in the English Football League (EFL).",
                "Sian Massey-Ellis works in the English Football League (EFL) and the Football Association (FA).",
                "Sian Massey-Ellis works in the United Kingdom.",
                "Massey-Ellis was born on March 2, 1983.",
                "Massey-Ellis was born in Liverpool, England.",
                "Massey-Ellis grew up with a strong passion.",
                "Massey-Ellis grew up with a strong passion for sports.",
                "Massey-Ellis grew up with a strong passion for sports, particularly football.",
                "Massey-Ellis completed her education.",
                "After completing her education, Massey-Ellis began her career.",
                "Massey-Ellis began her career in officiating.",
                "Massey-Ellis began her career in officiating in 2003.",
                "Massey-Ellis began her career in officiating at the local and grassroots level.",
                "Massey-Ellis began her career in officiating at the local and grassroots level in 2003.",
                "She quickly progressed through the ranks.",
                "She earned her place on the National List of Referees.",
                "She earned her place on the National List of Referees in 2008.",
                "In 2011, she became the first female referee.",
                "She became the first female referee in England.",
                "She took charge of a Football League match.",
                "A Football League match took place between Torquay United and Morecambe.",
                "She officiated the Football League match between Torquay United and Morecambe.",
                "She officiated a League Two fixture between Torquay United and Morecambe.",
                "The fixture between Torquay United and Morecambe was in the Football League.",
                "The fixture between Torquay United and Morecambe was in the League Two.",
                "Massey-Ellis has become a regular fixture.",
                "Massey-Ellis has become a regular fixture in the EFL.",
                "EFL is a league.",
                "Massey-Ellis officiates.",
                "Massey-Ellis officiates matches.",
                "Massey-Ellis officiates matches at various levels.",
                "Massey-Ellis officiates matches at various levels in the EFL.",
                "She has made history.",
                "She became the first female referee.",
                "She became the first female referee to officiate a Championship game.",
                "She became the first female referee to officiate a Championship game in 2014.",
                "She became the first female referee to officiate a League Cup match.",
                "She became the first female referee to officiate a League Cup match in 2015.",
                "Massey-Ellis has worked in the EFL.",
                "Massey-Ellis has also officiated.",
                "Massey-Ellis has also officiated in the FA Women's Super League.",
                "Massey-Ellis has also officiated in the FA Women's Championship.",
                "She has been a referee.",
                "She has been a referee for the FA Cup.",
                "She has been a referee for the FA WSL Continental Tyres Cup.",
                "She has been a referee for the FA Women's League Cup.",
                "Massey-Ellis has officiated internationally.",
                "Massey-Ellis has officiated matches in the UEFA Women's Champions League.",
                "Massey-Ellis has officiated matches in the UEFA European Women's Under-19 Championship.",
                "She was selected.",
                "She was selected to officiate.",
                "She was selected to officiate at the 2019 FIFA Women's World Cup.",
                "The 2019 FIFA Women's World Cup was held in France.",
                "Massey-Ellis has been recognized for her achievements.",
                "Massey-Ellis has been recognized for her achievements with numerous awards.",
                "Massey-Ellis received the FA Women's Football Award for Referee of the Year in 2014.",
                "Massey-Ellis received the FA Women's Football Award for Referee of the Year in 2015.",
                "Massey-Ellis received the FA Women's Football Award for Services to Refereeing in 2019.",
                "Massey-Ellis received the FA Women's Football Award for Services to Refereeing in 2019. (This fact is a repetition of the previous one, so it could be removed to make the list of facts non-redundant)",
                "She was named FA WSL Referee of the Year.",
                "The year was 2018.",
                "Massey-Ellis is an advocate.",
                "Massey-Ellis is an advocate off the field.",
                "Massey-Ellis works to promote equality.",
                "Massey-Ellis works to promote equality off the field.",
                "Massey-Ellis works to promote diversity.",
                "Massey-Ellis works to promote diversity off the field.",
                "Massey-Ellis works to promote equality and diversity.",
                "Massey-Ellis works to promote equality and diversity off the field.",
                "Football is a sport.",
                "Massey-Ellis is an advocate for women in sport.",
                "Massey-Ellis works to promote equality and diversity in football.",
                "She is a mentor.",
                "She is a mentor for the FA's NextGen programme.",
                "The FA's NextGen programme aims to encourage more women to become referees.",
                "She works with schools.",
                "She works with schools to inspire the next generation of footballers and officials.",
                "She works with community groups.",
                "She works with community groups to inspire the next generation of footballers and officials.",
            ]
        ]

        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
        
        self.config = EnvYAML(
            "configs/dedupsoft_configs.yaml",
            flatten=False,
            env_file="ENV.env",
            include_environment=True,
            OUTPUT_PATH="",
            SCORE_PATH="",
        ).export()['task']['scorer']
        self.config['decomposer']['sentence_level_checkworthy_scorer']['in_batch_num'] = 1

    # def test_factscore_decomposer(self):
    #     """
    #     """
        
    #     factscore_decomposer = FActScoreDecomposer(
    #         model_name="mistralai/Mistral-7B-Instruct-v0.2",
    #         base_url="http://localhost:9871/v1",
    #         api_key="token-abc123",
    #         example_path="factscore_decomp_examples.txt",
    #         sentencize=True
    #     )
        
    #     single_results = []
    #     for test_case in self.test_cases:
    #         results = [r.text for r in factscore_decomposer(test_case)]
    #         single_results.append(results)
            
    #     batch_results = factscore_decomposer(self.test_cases)
    #     batch_results = [[r.text for r in results] for results in batch_results]

    #     for single_result, batch_result in zip(single_results, batch_results):
    #         print('=' * 20)
    #         for br in batch_result:
    #             print(br)
    #         print('=' * 20)
    #         self.assertEqual(set(single_result), set(batch_result))
            
    # def test_checkworthy_scorer(self):
    #     """This is a bit trickier, where we dedup the outputs and see if the results
    #     are still equal.
    #     """
    #     sentence_level_checkworthy_scorer = LLMGeneralCheckWorthyScorer(
    #         model_name="mistralai/Mistral-7B-Instruct-v0.2",
    #         base_url="http://localhost:9871/v1",
    #         api_key="token-abc123",
    #         # This is very important to get consistent scoring.
    #         in_batch_num=1
    #     )

    #     # first score each instance individually
    #     single_result_dict = {}
    #     for group, case in zip(self.test_decomposed_cases_raw, self.test_cases):
    #         topic = case.topic
    #         instance_group = [ScorerInstance(text=raw_text, topic=topic) for raw_text in group]
    #         scores = sentence_level_checkworthy_scorer(instance_group, return_raw=False)
    #         for raw_text, score in zip(group, scores):
    #             single_result_dict[raw_text] = score
                
    #     # create scorer instances for the scorer
    #     flattened_cases = [ScorerInstance(text=raw_text, topic=case.topic) for group, case in zip(self.test_decomposed_cases_raw, self.test_cases) for raw_text in group]
    #     random_obj = Random(42)

    #     # Test with 10 random shuffle
    #     for _ in range(10):
    #         random_obj.shuffle(flattened_cases)
    #         batch_result = sentence_level_checkworthy_scorer(
    #             flattened_cases,
    #             return_raw=False
    #         )

    #         for case, score in zip(flattened_cases, batch_result):
    #             self.assertEqual(single_result_dict[case.text], score)
                
    # def test_claim_level_checkworthy_scorer(self):
    #     """We run the unli test_scorer to see if the results are consistent.
    #     """
        
    #     claim_level_checkworthy_scorer = UNLIConfidenceBoostScorer(
    #         bleached_templates=[
    #             "{topic} is a person.",
    #             "{topic} breathes.",
    #             "{topic} exists."
    #         ],
    #         entailer=SoftEntailer(
    #             model_name="Zhengping/roberta-large-unli",
    #             device="cuda:0",
    #             internal_batch_size=32,
    #             max_length=256,
    #         ),
    #     )
        
    #     # still run single_result dict
    #     single_result_dict = {}
    #     for group, case in zip(self.test_decomposed_cases_raw, self.test_cases):
    #         topic = case.topic
    #         instance_group = [ScorerInstance(text=raw_text, topic=topic) for raw_text in group]
    #         scores = claim_level_checkworthy_scorer(instance_group, return_raw=False)
    #         for raw_text, score in zip(group, scores):
    #             single_result_dict[raw_text] = score

    #     flattened_cases = [ScorerInstance(text=raw_text, topic=case.topic) for group, case in zip(self.test_decomposed_cases_raw, self.test_cases) for raw_text in group]
    #     random_obj = Random(42)
        
    #     for _ in range(10):
    #         random_obj.shuffle(flattened_cases)
    #         batch_result = claim_level_checkworthy_scorer(
    #             flattened_cases,
    #             return_raw=False
    #         )

    #         for case, score in zip(flattened_cases, batch_result):
    #             # self.assertEqual(single_result_dict[case.text], score)
    #             self.assertAlmostEqual(single_result_dict[case.text], score, delta=1e-5)
                
    # def test_factscore_scorer(self):
    #     """
    #     """
    #     # now we retrieve and compare the results using factscore
    #     # scorer, so this tests whether the retrieving and the check
    #     # gives the same results.
        
    #     scorer = LLMSupportScorer(
    #         model_name="mistralai/Mistral-7B-Instruct-v0.2",
    #         base_url="http://localhost:9871/v1",
    #         api_key="token-abc123",
    #         retriever_batch_size=256,
    #         db_path="db/enwiki-20230401.db",
    #         cache_dir=".cache/"
    #     )
        
        
    #     # still run single_result dict
    #     single_result_dict = {}
    #     for group, case in zip(self.test_decomposed_cases_raw, self.test_cases):
    #         topic = case.topic
    #         instance_group = [ScorerInstance(text=raw_text, topic=topic) for raw_text in group]
    #         scores = scorer(instance_group, return_raw=False)
    #         for raw_text, score in zip(group, scores):
    #             single_result_dict[raw_text] = score
                
    #     flattened_cases = [ScorerInstance(text=raw_text, topic=case.topic) for group, case in zip(self.test_decomposed_cases_raw, self.test_cases) for raw_text in group]
    #     random_obj = Random(42)
        
    #     for _ in range(10):
    #         random_obj.shuffle(flattened_cases)
    #         batch_result = scorer(
    #             flattened_cases,
    #             return_raw=False
    #         )

    #         for case, score in zip(flattened_cases, batch_result):
    #             # self.assertEqual(single_result_dict[case.text], score)
    #             self.assertAlmostEqual(single_result_dict[case.text], score, delta=1e-5)
    
    def test_unli_confidence_boost_scorer(self):
        """Take UNLI confidence scorer,
        and test if the results are consistent.
        """
        pass
                
    def test_deduplicated_decomposer(self):
        """We now test the deduplicated scorer, since
        now it is supposed to be the same for each new run
        """
        
        dedup_decomposer = DeduplicatedDecomposer(
            base_decomposer=FActScoreDecomposer(
                model_name="mistralai/Mistral-7B-Instruct-v0.2",
                example_path="factscore_decomp_examples.txt",
                sentencize=False,
                base_url="http://localhost:9871/v1",
                api_key="token-abc123"
            ),
            sentence_level_checkworthy_scorer=LLMGeneralCheckWorthyScorer(
                model_name="mistralai/Mistral-7B-Instruct-v0.2",
                base_url="http://localhost:9871/v1",
                api_key="token-abc123",
                # Again we need to set this to 1 to get consistent scoring.
                in_batch_num=1,
            ),
            claim_level_checkworthy_scorer=UNLIConfidenceBoostScorer(
                bleached_templates=[
                    "{topic} is a person.",
                    "{topic} breathes.",
                    "{topic} exists."
                ],
                entailer=SoftEntailer(
                    model_name="Zhengping/roberta-large-unli",
                    device="cuda:0",
                    internal_batch_size=32,
                    max_length=256,
                ),
            ),
            entailer=Entailer(
                model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                max_length=256,
                internal_batch_size=512,
            ),
            sentencize=True
        )
        
        single_result_dict = {}
        for test_case in self.test_cases:
            result = dedup_decomposer(test_case)
            single_result_dict[test_case.text] = set([r.text for r in result])
            
        # now we run the batched version
        random_obj = Random(42)
        test_cases = deepcopy(self.test_cases)
        for _ in range(5):
            random_obj.shuffle(test_cases)
            batch_results = dedup_decomposer(test_cases)
            
            for test_case, batch_result in zip(test_cases, batch_results):
                self.assertEqual(single_result_dict[test_case.text], set([r.text for r in batch_result]))
                
    # def test_retriever(self):
    #     """It seems that beforehand LLM is making different queries for support evaluation,
    #     which indicates that the retriever isn't equivalent under batched execution.
    #     """

    #     cache_dir = ".cache/"
    #     retriever = Retriever(
    #         db_path="db/enwiki-20230401.db",
    #         cache_path=os.path.join(cache_dir, "retriever-cache.json"),
    #         embed_cache_path=os.path.join(cache_dir, "retriever-embed-cache.pkl"),
    #         batch_size=256,
    #         device="cuda:0"
    #     )
        
    #     single_results = {}
    #     for test_case, claims in zip(self.test_cases, self.test_decomposed_cases_raw):
    #         topic = test_case.topic
    #         for claim in claims:
    #             passages = retriever.get_passages(topic, question=claim, k=5)
    #             single_results[claim] = passages

    #     flattened_cases = [ScorerInstance(text=raw_text, topic=case.topic) for group, case in zip(self.test_decomposed_cases_raw, self.test_cases) for raw_text in group]
    #     random_obj = Random(42)
    #     for _ in range(10):
    #         del retriever
    #         retriever = Retriever(
    #             db_path="db/enwiki-20230401.db",
    #             cache_path=os.path.join(cache_dir, "retriever-cache.json"),
    #             embed_cache_path=os.path.join(cache_dir, "retriever-embed-cache.pkl"),
    #             batch_size=256,
    #             device="cuda:0"
    #         )
    #         random_obj.shuffle(flattened_cases)
    #         batch_results = retriever.get_passages_batched(topics=[instance.topic for instance in flattened_cases], questions=[instance.text for instance in flattened_cases], k=5)
    #         for case, passages in zip(flattened_cases, batch_results):
    #             single_passages = single_results[case.text]
    #             self.assertEqual(len(passages), len(single_passages))
    #             for passage, single_passage in zip(passages, single_passages):
    #                 self.assertDictEqual(passage, single_passage)
                
    # def test_decompose_scorer_without_dedup(self):
    #     """If the decomposer works, then it logically follows that
    #     the deduplicated scorer should also work in
    #     batch as well.
    #     """
        
    #     decomp_scorer = DecomposeScorer(
    #         abstention_detector=FActScoreAbstentionDetector(),
    #         decomposer=FActScoreDecomposer(
    #             model_name="mistralai/Mistral-7B-Instruct-v0.2",
    #             example_path="factscore_decomp_examples.txt",
    #             sentencize=True,
    #             base_url="http://localhost:9871/v1",
    #             api_key="token-abc123"
    #         ),
    #         base_scorer=LLMSupportScorer(
    #             model_name="mistralai/Mistral-7B-Instruct-v0.2",
    #             base_url="http://localhost:9871/v1",
    #             api_key="token-abc123",
    #             retriever_batch_size=256,
    #             db_path="db/enwiki-20230401.db",
    #             cache_dir=".cache/"
    #         ),
    #         aggregator=FActScoreAggregator(gamma=10)
    #     )
        
    #     single_result_dict = {}
    #     for test_case in self.test_cases:
    #         result = decomp_scorer(test_case, return_raw=False)
    #         single_result_dict[test_case.text] = result
            
    #     # now we run the batched version
    #     random_obj = Random(42)
    #     test_cases = deepcopy(self.test_cases)
        
    #     for _ in range(10):
    #         random_obj.shuffle(test_cases)
    #         batch_results = decomp_scorer(test_cases, return_raw=False)
            
    #         for test_case, batch_result in zip(test_cases, batch_results):
    #             self.assertAlmostEqual(single_result_dict[test_case.text], batch_result, delta=1e-5)
                
    # def test_decompose_scorer_with_dedup(self):
    #     """Now we test with the final target, where the deduplicator is the
    #     same within our configuration of deduplicator.
    #     """
        
    #     # decomp_scorer = DecomposeScorer(
    #     #     abstention_detector=FActScoreAbstentionDetector(),
    #     #     decomposer=DeduplicatedDecomposer(
    #     #             base_decomposer=FActScoreDecomposer(
    #     #             model_name="mistralai/Mistral-7B-Instruct-v0.2",
    #     #             example_path="factscore_decomp_examples.txt",
    #     #             # We do observe that the sentencize, if not applied,
    #     #             # will sometimes lead llm to generate outputs that can
    #     #             # fail the parsing pipeline, so if not wrapped by another pipeline
    #     #             # that does sentencize, it is highly recommended to sentencize
    #     #             sentencize=False,
    #     #             base_url="http://localhost:9871/v1",
    #     #             api_key="token-abc123"
    #     #         ),
    #     #         sentence_level_checkworthy_scorer = LLMGeneralCheckWorthyScorer(
    #     #             model_name="mistralai/Mistral-7B-Instruct-v0.2",
    #     #             base_url="http://localhost:9871/v1",
    #     #             api_key="token-abc123",
    #     #             # This is very important to get consistent scoring.
    #     #             in_batch_num=1
    #     #         ),
    #     #         claim_level_checkworthy_scorer=UNLIConfidenceBoostScorer(
    #     #             bleached_templates=[
    #     #                 "{topic} is a person.",
    #     #                 "{topic} breathes.",
    #     #                 "{topic} exists."
    #     #             ],
    #     #             entailer=SoftEntailer(
    #     #                 model_name="Zhengping/roberta-large-unli",
    #     #                 device="cuda:0",
    #     #                 internal_batch_size=32,
    #     #                 max_length=256,
    #     #             ),
    #     #         ),
    #     #         entailer=Entailer(
    #     #             model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    #     #             device="cuda:0",
    #     #             internal_batch_size=512,
    #     #             max_length=256,
    #     #         ),
    #     #         sentencize=True
    #     #     ),
    #     #     base_scorer=LLMSupportScorer(
    #     #         model_name="mistralai/Mistral-7B-Instruct-v0.2",
    #     #         base_url="http://localhost:9871/v1",
    #     #         api_key="token-abc123",
    #     #         retriever_batch_size=256,
    #     #         db_path="db/enwiki-20230401.db",
    #     #         cache_dir=".cache/"
    #     #     ),
    #     #     aggregator=FActScoreAggregator(gamma=10)
    #     # )
        
    #     decomp_scorer = Scorer.from_params(self.config)
        
    #     cached = [1.0, 1.0, 1.0, .75, 0.6388888888888888]
        
    #     single_result_dict = {}
    #     for tidx, test_case in enumerate(self.test_cases):
    #         # result = decomp_scorer(test_case, return_raw=False)
    #         single_result_dict[test_case.text] = cached[tidx]
            
    #     # now we run the batched version
    #     random_obj = Random(42)
    #     test_cases = deepcopy(self.test_cases)
        
    #     for _ in range(2):
    #         random_obj.shuffle(test_cases)
    #         batch_results = decomp_scorer(test_cases, return_raw=False)
    #         # print(batch_results)
            
    #         for test_case, batch_result in zip(test_cases, batch_results):
    #             # print(test_case, batch_result)
    #             self.assertAlmostEqual(single_result_dict[test_case.text], batch_result, delta=1e-5)