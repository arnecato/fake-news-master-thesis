from sklearn.metrics.pairwise import cosine_similarity
import spacy
from abc import ABC, abstractmethod
import numpy as np
import time


# TODO: always use spaCy as sentence extractor and allow any other model to be used for quick word vector lookup (spaCy, FastText, Glove, Word2Vec, Google News etc)

class VectorFactory(ABC):
    def __init__(self, model, dim):
        """
        Initialize the VectorFactory with a pre-trained model and the desired dimension of the vectors.
        
        Parameters:
        model: The pre-trained word embedding model (e.g., FastText, GloVe).
        dim: The dimension of the vectors (int).
        """
        self.model = model
        self.dim = dim

    @abstractmethod
    def word_vector(self, word):
        """
        Generate the vector for a single word.
        
        Parameters:
        word: The word to generate the vector for (str).
        
        Returns:
        A vector representing the word (list of floats).
        """
        pass

    @abstractmethod
    def sentence_vector(self, sentence):
        """
        Generate the vector for a sentence.
        
        Parameters:
        sentence: The sentence to generate the vector for (str).
        
        Returns:
        A vector representing the sentence (list of floats).
        """
        pass

    @abstractmethod
    def document_vector(self, sentences):
        """
        Generate the vector for a document.
        
        Parameters:
        sentences: A list of sentences that make up the document (list of str).
        
        Returns:
        A vector representing the document (list of floats).
        """
        pass

class SpacyVectorFactory(VectorFactory):
    def __init__(self, model):
        """
        Initialize the SpacyVectorFactory with a pre-trained spaCy model.
        
        Parameters:
        model: The spaCy pre-trained word embedding model.
        """
        config = {"punct_chars": ['!', '.', '?', ';', '—']} 
        '''['!', '.', '?', '։', '؟', '۔', '܀', '܁', '܂', '߹', '।', '॥', '၊', '။', '።',
                '፧', '፨', '᙮', '᜵', '᜶', '᠃', '᠉', '᥄', '᥅', '᪨', '᪩', '᪪', '᪫',
                '᭚', '᭛', '᭞', '᭟', '᰻', '᰼', '᱾', '᱿', '‼', '‽', '⁇', '⁈', '⁉',
                '⸮', '⸼', '꓿', '꘎', '꘏', '꛳', '꛷', '꡶', '꡷', '꣎', '꣏', '꤯', '꧈',
                '꧉', '꩝', '꩞', '꩟', '꫰', '꫱', '꯫', '﹒', '﹖', '﹗', '！', '．', '？',
                '𐩖', '𐩗', '𑁇', '𑁈', '𑂾', '𑂿', '𑃀', '𑃁', '𑅁', '𑅂', '𑅃', '𑇅',
                '𑇆', '𑇍', '𑇞', '𑇟', '𑈸', '𑈹', '𑈻', '𑈼', '𑊩', '𑑋', '𑑌', '𑗂',
                '𑗃', '𑗉', '𑗊', '𑗋', '𑗌', '𑗍', '𑗎', '𑗏', '𑗐', '𑗑', '𑗒', '𑗓',
                '𑗔', '𑗕', '𑗖', '𑗗', '𑙁', '𑙂', '𑜼', '𑜽', '𑜾', '𑩂', '𑩃', '𑪛',
                '𑪜', '𑱁', '𑱂', '𖩮', '𖩯', '𖫵', '𖬷', '𖬸', '𖭄', '𛲟', '𝪈', '｡', '。']}'''
        model.add_pipe("sentencizer") #, config=config, before="tok2vec")
        super().__init__(model, model.vocab.vectors_length)

    def word_vector(self, word):
        """
        Generate the vector for a single word using spaCy.
        
        Parameters:
        word: The word to generate the vector for (str).
        
        Returns:
        A vector representing the word (numpy array).
        """
        token = self.model(word)
        if token.has_vector:
            return token.vector
        else:
            return [0.0] * self.dim

    def sentence_vector(self, sentence):
        """
        Generate the vector for a sentence by averaging the vectors of the words in the sentence using spaCy.
        
        Parameters:
        sentence: The sentence to generate the vector for (str).
        
        Returns:
        A vector representing the sentence (numpy array).
        """
        doc = self.model(sentence)
        vectors = [token.vector for token in doc if token.has_vector]
        if len(vectors) == 0:
            return [0.0] * self.dim
        # Average the word vectors to get the sentence vector
        sentence_vector = sum(vectors) / len(vectors)
        return sentence_vector

    def document_vector(self, text):
        """
        Generate the vector for a document by splitting the text into sentences and 
        averaging the vectors of the sentences in the document using spaCy.
        
        Parameters:
        text: The text of the document (str).
        
        Returns:
        A vector representing the document (numpy array).
        """
        # Use spaCy to split the text into sentences
        doc = self.model(text)

        # Generate sentence vectors and average them
        sentence_vectors = [sent.vector for sent in doc.sents] 
        if len(sentence_vectors) == 0:
            return [0.0] * self.dim
        
        # Average the sentence vectors to get the document vector
        document_vector = sum(sentence_vectors) / len(sentence_vectors)
        return document_vector
    
def main():
    # load spacy and speed up by disabling unnecessary components
    model = spacy.load("en_core_web_lg", disable=["parser", "ner", "tagger", "parser", "attribute_ruler", "lemmatizer"])  # _lg
    vfac = SpacyVectorFactory(model)
    text = "As U.S. budget fight looms, Republicans flip their fiscal script. WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a “fiscal conservative” on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBS’ “Face the Nation,” drew a hard line on federal spending, which lawmakers are bracing to do battle over in January. When they return from the holidays on Wednesday, lawmakers will begin trying to pass a federal budget in a fight likely to be linked to other issues, such as immigration policy, even as the November congressional election campaigns approach in which Republicans will seek to keep control of Congress. President Donald Trump and his Republicans want a big budget increase in military spending, while Democrats also want proportional increases for non-defense “discretionary” spending on programs that support education, scientific research, infrastructure, public health and environmental protection. “The (Trump) administration has already been willing to say: ‘We’re going to increase non-defense discretionary spending ... by about 7 percent,’” Meadows, chairman of the small but influential House Freedom Caucus, said on the program. “Now, Democrats are saying that’s not enough, we need to give the government a pay raise of 10 to 11 percent. For a fiscal conservative, I don’t see where the rationale is. ... Eventually you run out of other people’s money,” he said. Meadows was among Republicans who voted in late December for their party’s debt-financed tax overhaul, which is expected to balloon the federal budget deficit and add about $1.5 trillion over 10 years to the $20 trillion national debt. “It’s interesting to hear Mark talk about fiscal responsibility,” Democratic U.S. Representative Joseph Crowley said on CBS. Crowley said the Republican tax bill would require the  United States to borrow $1.5 trillion, to be paid off by future generations, to finance tax cuts for corporations and the rich. “This is one of the least ... fiscally responsible bills we’ve ever seen passed in the history of the House of Representatives. I think we’re going to be paying for this for many, many years to come,” Crowley said. Republicans insist the tax package, the biggest U.S. tax overhaul in more than 30 years,  will boost the economy and job growth. House Speaker Paul Ryan, who also supported the tax bill, recently went further than Meadows, making clear in a radio interview that welfare or “entitlement reform,” as the party often calls it, would be a top Republican priority in 2018. In Republican parlance, “entitlement” programs mean food stamps, housing assistance, Medicare and Medicaid health insurance for the elderly, poor and disabled, as well as other programs created by Washington to assist the needy. Democrats seized on Ryan’s early December remarks, saying they showed Republicans would try to pay for their tax overhaul by seeking spending cuts for social programs. But the goals of House Republicans may have to take a back seat to the Senate, where the votes of some Democrats will be needed to approve a budget and prevent a government shutdown. Democrats will use their leverage in the Senate, which Republicans narrowly control, to defend both discretionary non-defense programs and social spending, while tackling the issue of the “Dreamers,” people brought illegally to the country as children. Trump in September put a March 2018 expiration date on the Deferred Action for Childhood Arrivals, or DACA, program, which protects the young immigrants from deportation and provides them with work permits. The president has said in recent Twitter messages he wants funding for his proposed Mexican border wall and other immigration law changes in exchange for agreeing to help the Dreamers. Representative Debbie Dingell told CBS she did not favor linking that issue to other policy objectives, such as wall funding. “We need to do DACA clean,” she said.  On Wednesday, Trump aides will meet with congressional leaders to discuss those issues. That will be followed by a weekend of strategy sessions for Trump and Republican leaders on Jan. 6 and 7, the White House said. Trump was also scheduled to meet on Sunday with Florida Republican Governor Rick Scott, who wants more emergency aid. The House has passed an $81 billion aid package after hurricanes in Florida, Texas and Puerto Rico, and wildfires in California. The package far exceeded the $44 billion requested by the Trump administration. The Senate has not yet voted on the aid. "
    text2 = "Federal judge partially lifts Trump's latest refugee restrictions. WASHINGTON (Reuters) - A federal judge in Seattle partially blocked U.S. President Donald Trump’s newest restrictions on refugee admissions on Saturday, the latest legal defeat for his efforts to curtail immigration and travel to the United States. The decision by U.S. District Judge James Robart is the first judicial curb on rules the Trump administration put into place in late October that have contributed significantly to a precipitous drop in the number of refugees being admitted into the country. Refugees and groups that assist them argued in court that the administration’s policies violated the Constitution and federal rulemaking procedures, among other claims. Department of Justice attorneys argued in part that U.S. law grants the executive branch the authority to limit refugee admissions in the way that it had done so. On Oct. 24, the Trump administration effectively paused refugee admissions from 11 countries mostly in the Middle East and Africa, pending a 90-day security review, which was set to expire in late January. The countries subject to the review are Egypt, Iran, Iraq, Libya, Mali, North Korea, Somalia, South Sudan, Sudan, Syria and Yemen. For each of the last three years, refugees from the 11 countries made up more than 40 percent of U.S. admissions. A Reuters review of State Department data showed that as the review went into effect, refugee admissions from the 11 countries plummeted. Robart ruled that the administration could carry out the security review, but that it could not stop processing or admitting refugees from the 11 countries in the meantime, as long as those refugees have a “bona fide” connection to the United States. As part of its new restrictions, the Trump administration had also paused a program that allowed for family reunification for refugees, pending further security screening procedures being put into place. Robart ordered the government to re-start the program, known as “follow-to-join”. Approximately 2,000 refugees were admitted into the United States in fiscal year 2015 under the program, according to Department of Homeland Security data. Refugee advocacy groups praised Robart’s decision.  “This ruling brings relief to thousands of refugees in precarious situations in the Middle East and East Africa, as well as to refugees already in the U.S. who are trying to reunite with their spouses and children,” said Mariko Hirose, litigation director for the International Refugee Assistance Project, one of the plaintiffs in the case. A Justice Department spokeswoman, Lauren Ehrsam, said the department disagrees with Robart’s ruling and is “currently evaluating the next steps”. Robart, who was appointed to the bench by Republican former President George W. Bush, emerged from relative obscurity in February, when he issued a temporary order to lift the first version of Trump’s travel ban. On Twitter, Trump called him a “so-called judge” whose “ridiculous” opinion “essentially takes law-enforcement away from our country”. Robart’s ruling represented the second legal defeat in two days for the Trump administration. On Friday, a U.S. appeals court said Trump’s travel ban targeting six Muslim-majority countries should not be applied to people with strong U.S. ties, but said its ruling would be put on hold pending a decision by the U.S. Supreme Court. " #"Utter bullshit. Utter bullshit. Utter bullshit. Utter bullshit. Utter bullshit.  Trump. Conservative. Thing thang. Democrats. Republicans.  Trump. Conservative. Thing thang. Democrats. Republicans.  Trump. Conservative. Thing thang. Democrats. Republicans.  Trump. Conservative. Thing thang. Democrats. Republicans.  Trump. Conservative. Thing thang. Democrats. Republicans.  Trump. Conservative. Thing thang. Democrats. Republicans.  Trump. Conservative. Thing thang. Democrats. Republicans.  Trump. Conservative. Thing thang. Democrats. Republicans."
    time0 = time.perf_counter()
    #b = model.vocab['budget'].vector
    doc_v1 = vfac.document_vector(text)
    doc_v2 = vfac.document_vector(text2)
    #print(cosine_similarity([doc_v1], [doc_v2]), np.linalg.norm(doc_v1 - doc_v2))
    print(time.perf_counter() - time0)
    

if __name__ == '__main__':
    main()