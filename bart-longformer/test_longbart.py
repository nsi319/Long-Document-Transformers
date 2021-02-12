from transformers import BartTokenizer,HfArgumentParser

from modeling_bart import BartForConditionalGeneration

from modeling_longbart import LongBartForConditionalGeneration

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class TestArguments:
    """
    Arguments pertaining to testing of model.
    """
    model_path: str = field(
        metadata={
        "help":'The name or path of the model you want to test'
        },
    )

def main():
    parser = HfArgumentParser(TestArguments)
    args = parser.parse_args_into_dataclasses()[0]

    model = LongBartForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer = BartTokenizer.from_pretrained(args.model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    padding = "max_length" 

    text="""The Securities and Exchange Commission charged William "Bill" Bowser, Christopher Ashby, Scott Beynon, and Jordan Nelson with securities fraud for misappropriating investor funds meant for the development and construction of new event centers. The SEC's complaint alleges that from approximately January 2017 to February 2019, Ashby, Beynon, and Nelson, through entities they controlled, sold investors interests in for-profit event centers purportedly being developed by Noah Corporation, an entity Bowser controlled. As alleged, Bowser diverted investor funds earmarked for specific properties and instead used them for Noah Corporation's and Bowser's operational and other expenses and to pay prior investors, rather than for construction of event centers as represented to investors. The complaint further alleges that contrary to their representations to investors, Ashby, Beynon, and Nelson failed to escrow investor funds and disbursed them to an entity Bowser controlled without having any controls in place to ensure that the disbursements were for legitimate expenses. The SEC's complaint, filed in federal district court in Utah, charges Bowser with violating the antifraud provisions of Sections 17(a)(1) and 17(a)(3) of the Securities Act of 1933 and Section 10(b) of the Securities Exchange Act of 1934 and Rules 10b-5(a) and (c) thereunder. The complaint charges Ashby, Beynon, and Nelson with violating the antifraud provisions of Sections 17(a)(2) and 17(a)(3) of the Securities Act and the registration provisions of Section 15(a) of the Exchange Act. Without admitting or denying the allegations of the complaint, the defendants have consented to judgments enjoining them from violating the charged provisions, ordering Bowser to pay disgorgement of $47,796 with prejudgment interest of $6,402 and a $192,768 penalty, ordering Ashby to pay disgorgement of $551,161 with prejudgment interest of $43,994 and a $96,384 penalty, ordering Beynon to pay disgorgement of $585,426 with prejudgment interest of $46,729 and a $96,384 penalty, and ordering Nelson to pay disgorgement of $281,273 with prejudgment interest of $22,451 and a $96,384 penalty. The SEC's investigation was conducted by Cheryl Mori and was supervised by Daniel Wadley and Amy Oliver of the Salt Lake Regional Office. The litigation will be led by Casey Fronk. The Securities and Exchange Commission today announced charges against DeAndre P. Sears and MASears LLC d/b/a Picasso Group, an entity he controlled and operated, with registration violations for unlawfully selling securities of Florida-based real estate firm EquiAlt LLC to retail investors. The SEC previously filed an enforcement action against EquiAlt LLC, its CEO Brian Davison, and its Managing Director Barry Rybicki on February 11, 2020, in connection with the alleged scheme. According to the SEC's complaint, between 2014 and 2020, Sears directly and indirectly, through the use of third-party agents, sold at least $25 million of EquiAlt's securities to more than 145 largely unaccredited, unsophisticated, and elderly retail investors located in 25 states. During that period, Sears was identified in EquiAlt private placement memoranda as Managing Director of Investments, President of Business Development and Marketing, or Vice President of Investor Relations. Sears, through Picasso Group, received approximately $3.5 million in transaction-based sales commissions from EquiAlt, despite neither being registered as broker dealers. The complaint alleges that beginning in approximately 2016, EquiAlt was actually operating a Ponzi scheme during which it raised more than $170 million from approximately 1,100 investors in 35 states. The SEC's complaint charges Sears and Picasso Group with violating the securities registration provisions of Sections 5(a) and 5(c) of the Securities Act of 1933, and the broker-dealer registration provisions of Section 15(a)(1) of the Securities Exchange Act of 1934. Without admitting or denying the allegations in the complaint, Sears and Picasso Group have agreed to the entry of a judgment providing injunctive relief with disgorgement and civil penalties to be determined by a court at a later date. Sears also agreed to associational and penny stock bars as part of a settled follow-on administrative proceeding. The SEC's continuing investigation is being conducted by Chanel T. Rowe and Andre Zamorano, with assistance from Mark Dee, and supervised by Thierry Olivier Desmet and Glenn S. Gordon in the Miami Regional Office. The SEC's litigation is being led by Alise Johnson and supervised by Andrew O. Schiff. The SEC encourages investors to check the backgrounds of people selling investments by using the SEC's Investor.gov to identify quickly whether they are registered professionals and confirm their identity. On December 30, 2020, the U. S. District Court for the Southern District of New York entered final judgments against two penny stock promoters whom the Commission charged in connection with several alleged pump-and-dump schemes involving stocks they were touting in their supposedly independent newsletters. The SEC's complaint in this action, filed in November 2014, alleged that Anthony Thompson, Jr., Jay Fung, and a third defendant, Eric Van Nguyen, worked in concert to gain control of a large portion of shares in the stock of microcap companies, then hyped those stocks in newsletters they distributed to prospective investors. According to the complaint, the newsletters published by Thompson, Fung, and Van Nguyen misleadingly stated that they "may" or "might" sell shares they owned when in reality they always intended to sell -- and in some instances already were selling - the stocks they were promoting. As alleged, they also failed to fully disclose in their newsletters the amounts of compensation they were receiving for promoting the stocks. Thompson and Fung, who were both previously convicted in a parallel state criminal case for conduct that was the subject of the SEC's action, each consented to the entry of a final judgment enjoining them from future violations of the anti-touting provisions of Section 17(b) of the Securities Act of 1933. Thompson further consented to pay disgorgement of $624,882 plus prejudgment interest of $137,381, and Fung consented to pay disgorgement of $1,766,083 plus prejudgment interest of $244,308. Fung's obligation to pay disgorgement and prejudgment interest will be deemed satisfied by the $2,800,000 he was ordered to pay pursuant to an order of restitution entered in a related criminal action filed by the Manhattan District Attorney's Office. The SEC's litigation continues with respect to Van Nguyen. The SEC's litigation has been handled by Peter Pizzani, Mark R. Sylvester, and Thomas P. Smith, Jr., and supervised by Sanjay Wadhwa, all of the New York Regional Office."""
    input_tokenized = tokenizer.encode(text, return_tensors='pt',padding=padding,pad_to_max_length=True, padding_side='right', max_length=4096,truncation=True).to(device)
    
    
    dim = list(input_tokenized.size())
    print("Input Shape: {} (padded to max_length)".format(dim))

    summary_ids = model.generate(input_tokenized,
                                      num_beams=4,
                                      no_repeat_ngram_size=3,
                                      length_penalty=1.0,
                                      min_length=512,
                                      max_length=2016)
    
    dim = list(summary_ids.size())
    print("Summary Shape: ", dim)
      
    summ = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
    
    print("Text: ")
    print(text)
    print("Summary: ")
    print(summ)

if __name__ == "__main__":
    main()