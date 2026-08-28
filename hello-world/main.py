from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash-lite"


def main():
    information = """
        Khalid ibn al-Walid ibn al-Mughira al-Makhzumi[a] (died 642) was a 7th-century Arab military commander. He initially led campaigns against Muhammad on behalf of the Quraysh, but later converted to Islam and spent the remainder of his career serving Muhammad and the first two Rashidun caliphs Abu Bakr and Umar as a commander of the Muslim army. Khalid played leading command roles in the Ridda Wars against rebel tribes in Arabia in 632–633, the initial campaigns in Sasanian Iraq in 633–634, and the conquest of Byzantine Syria in 634–638.

        As a horseman of the Quraysh's aristocratic Banu Makhzum clan, which ardently opposed Muhammad, Khalid played an instrumental role in defeating Muhammad and his followers during the Battle of Uhud in 625. In 627 or 629, he converted to Islam in the presence of Muhammad, who inducted him as an official military commander among the Muslims and gave him the title Sayf Allah (lit. 'Sword of God'). During the Battle of Mu'ta, Khalid coordinated the safe withdrawal of Muslim troops against the Byzantines. He also led the Bedouins under the Muslim army during the Muslim conquest of Mecca in 629–630 and the Battle of Hunayn in 630. After Muhammad's death, Khalid was appointed to Najd and al-Yamama to suppress or subjugate the Arab tribes opposed to the nascent Muslim state; this campaign culminated in Khalid's victories over the rebel leaders Tulayha at the Battle of Buzakha in September 632 and Musaylima at the Battle of al-Yamama in December 632.

        Khalid subsequently launched campaigns against predominantly Christian Arab tribes and the Sasanian Persian garrisons along the Euphrates valley in Iraq. Abu Bakr later reassigned him to command the Muslim armies in Syria, where he led his forces on an unconventional march across a long, waterless stretch of the Syrian desert, boosting his reputation as a military strategist. As a result of decisive victories led by Khalid against the Byzantines at Ajnadayn (634), Fahl (634 or 635), Damascus (634–635), and the Yarmouk (636), the Muslim army conquered most of the Levant. Khalid was subsequently demoted and removed from the army's high command by Umar. Khalid continued service as the key lieutenant of his successor Abu Ubayda ibn al-Jarrah in the siege of Homs, Aleppo, and the Battle of Qinnasrin, all in 637–638. These engagements collectively precipitated the retreat of imperial Byzantine troops from Syria under Emperor Heraclius. Around 638, Umar dismissed Khalid from both his military command and his position as governor of Qinnasrin. Khalid died in 642 either in Medina or Homs.

        Khalid is generally considered by historians to be one of the most seasoned and accomplished generals in Islamic history, and he is likewise commemorated throughout the Arab world. Islamic tradition credits him with decisive battlefield tactics and effective leadership during the early Muslim conquests. However, historical accounts offer differing perspectives on certain events, including his execution of Malik ibn Nuwayra during the Ridda Wars and his dismissal from command by Umar. Khalid's military fame disturbed some pious early Muslims, most notably Umar, who feared it could develop into a personality cult. In Sunni tradition, Khalid is generally honored as a heroic figure, whereas Shia tradition portrays him more critically.
    """

    summary_template = """
    given the information {information} about a person I want to create: 
    1. A short summary
    2. Two facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGoogleGenerativeAI(model=MODEL)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
