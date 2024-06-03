"""Run test on the dedupsafe decomposer which is to apply dedupsoft
on top of SAFE decomposition pipeline.
"""

from copy import deepcopy
from unittest import TestCase
from src.decomposer import Decomposer
from src.utils.instances import ScorerInstance


class TestDedupSafeDecomposer(TestCase):
    def setUp(self):
        """
        """
        from envyaml import EnvYAML

        self.config = EnvYAML(
            "configs/dedupsafe_configs.yaml",
            flatten=False,
            env_file="ENV.env",
            include_environment=True,
            OUTPUT_PATH="",
            SCORE_PATH="",
        ).export()['task']['scorer']['decomposer']

        self.decomposer: Decomposer = Decomposer.from_params(deepcopy(self.config))
        
        self.resources = [
            "Kalki Koechlin is an Indian actress and writer known for her work in Hindi films. Born on January 10, 1984, in Pondicherry, India, Kalki spent her early years in Auroville before moving to London to study drama and theatre. She made her acting debut in the critically acclaimed film \"Dev.D\" in 2009, for which she received the Filmfare Award for Best Supporting Actress.\n\nKalki has since appeared in a variety of films, showcasing her versatility as an actress. Some of her notable performances include \"Zindagi Na Milegi Dobara,\" \"Margarita with a Straw,\" and \"Gully Boy.\" She has garnered praise for her unconventional choice of roles and her ability to portray complex characters with depth and authenticity.\n\nIn addition to her acting career, Kalki is also a talented writer and has written for various publications on topics such as feminism, mental health, and social issues. She is known for her outspoken views on gender equality and has been a vocal advocate for women's rights in India. Kalki Koechlin continues to be a prominent figure in the Indian film industry, known for her bold and fearless approach to her craft.",
            "Adil Rami is a professional French footballer who was born on December 27, 1985, in Bastia, France. He primarily plays as a central defender and is known for his physicality, aerial prowess, and strong defensive abilities. Rami began his professional career at Lille OSC in 2006 and quickly established himself as a key player in the team's defense.\n\nIn 2011, Rami made a move to Spanish club Valencia CF, where he continued to impress with his performances. He later joined Italian giants AC Milan in 2014 before returning to France to play for Marseille in 2017. Rami has also represented the French national team, earning caps and playing a key role in their success at the 2018 FIFA World Cup, where they emerged as champions.\n\nOff the field, Rami is known for his charismatic personality and has gained a significant following on social media. He is also involved in various charitable activities and has shown a commitment to giving back to the community. Rami's career has been marked by success at both the club and international levels, making him one of the prominent figures in French football.",
            "Song Kang is a South Korean actor born on April 23, 1994. He gained popularity for his roles in various Korean dramas such as \"Love Alarm,\" \"Navillera,\" and \"Nevertheless.\" Song Kang began his acting career in 2017 and quickly rose to fame for his charming looks and versatile acting skills. He is known for his ability to portray a wide range of characters, from sweet and romantic to intense and brooding. With his rising popularity both in South Korea and internationally, Song Kang has become one of the most sought-after young actors in the Korean entertainment industry.",
            " Rakhi Sawant is a renowned Indian actress, model, dancer, and television personality. She was born on November 27, 1978, in Mumbai, Maharashtra, India. Rakhi began her career in the entertainment industry as a contestant on the reality show \"Bigg Boss\" in 2006. Since then, she has appeared in numerous Bollywood films, reality shows, and television series.\n\nRakhi made her acting debut in the film \"Agasthya\" in 2004, and her first major role was in the film \"Masti\" in 2004. She has since appeared in several other films, including \"Shararat,\" \"Krishna Cottage,\" \"Main Hoon Na,\" and \"Bhootwala Serial Killer.\"\n\nApart from acting, Rakhi has also been a popular contestant on various reality shows, including \"Nach Baliye,\" \"Jhalak Dikhhla Jaa,\" and \"Fear Factor.\" She has also hosted several television shows, including \"Rakhi Ka Swayamvar,\" \"Rakhi Ka Insaaf,\" and \"Rakhi Ki Salaam-e-Ishq.\"\n\nRakhi is known for her bold and controversial personality, and she has been a subject of media attention for her public appearances and statements. She has also been involved in several controversies throughout her career, including her alleged relationship with actor Abhishek Aasman Kadam and her participation in a reality show where she claimed to be married to an American businessman.\n\nDespite the controversies, Rakhi continues to be a popular figure in the Indian entertainment industry. She has a large following on social media and is known for her outspoken nature and her ability to entertain her fans. She is also an active philanthropist and has worked with several NGOs to raise awareness about various social issues.",
            "Sian Massey-Ellis is a highly respected and accomplished sports official, best known for her work as a referee in the English Football League (EFL) and the Football Association (FA) in the United Kingdom. Born on March 2, 1983, in Liverpool, England, Massey-Ellis grew up with a strong passion for sports, particularly football.\n\nAfter completing her education, Massey-Ellis began her career in officiating in 2003, starting with local and grassroots football. She quickly progressed through the ranks, earning her place on the National List of Referees in 2008. In 2011, she became the first female referee to officiate a Football League match in England, taking charge of a League Two fixture between Torquay United and Morecambe.\n\nMassey-Ellis has since become a regular fixture in the EFL, officiating numerous matches at various levels. She has also made history by becoming the first female referee to officiate a Championship game in 2014, as well as the first female referee to officiate a League Cup match in 2015.\n\nIn addition to her work in the EFL, Massey-Ellis has also officiated in the FA Women's Super League and the FA Women's Championship. She has been a referee for the FA Cup and the FA WSL Continental Tyres Cup, as well as the FA Women's League Cup.\n\nMassey-Ellis has also officiated internationally, including matches in the UEFA Women's Champions League and the UEFA European Women's Under-19 Championship. She was also selected to officiate at the 2019 FIFA Women's World Cup in France.\n\nMassey-Ellis has been recognized for her achievements with numerous awards, including the FA Women's Football Award for Referee of the Year in 2014 and 2015, and the FA Women's Football Award for Services to Refereeing in 2019. She was also named the FA WSL Referee of the Year in 2018.\n\nOff the field, Massey-Ellis is an advocate for women in sport and works to promote equality and diversity in football. She is a mentor for the FA's NextGen programme, which aims to encourage more women to become referees, and she also works with schools and community groups to inspire the next generation of footballers and officials.",
        ]

        self.test_cases = [
            ScorerInstance(
                text=self.resources[0],
                topic="Kalki Koechlin", source_text=self.resources[0]),
            ScorerInstance(
                text=self.resources[1],
                topic="Adil Rami", source_text=self.resources[1]),
            ScorerInstance(
                text=self.resources[2],
                topic="Song Kang", source_text=self.resources[2]),
            ScorerInstance(
                text=self.resources[3],
                topic="Rakhi Sawant", source_text=self.resources[3]
            ),
            ScorerInstance(
                text=self.resources[4],
                topic="Sian Massey-Ellis", source_text=self.resources[4]
            )
        ]
        
    def test_safe_decomposer(self):
        decomposed = self.decomposer(self.test_cases)
        
        for dcp in decomposed:
            print("=" * 20)
            for idx, item in enumerate(dcp):
                print(f"{idx:02d}. {item.text}")
            print("=" * 20)