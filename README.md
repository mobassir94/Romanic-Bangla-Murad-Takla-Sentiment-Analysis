# Romanic-Bangla-Murad-Takla-Sentiment-Analysis
in this repo,i will share how to fine tune xlm roberta large for efficiently solving complex multilingual sentiment analysis task like romanic bangla and murad takla (language distortion) dataset

# what is murad takla language and why do we care?

The phrase “Kala haba”, may be categorized as incorrect, is quite a well known phrase to netizens  of Bangladesh. Although , ideally it shouldn’t mean anything(or mean something completely unrelated to the meaning that it was originally supposed to mean ), a general understanding is still possible because of our familiarity of "বাংলা" language. This familiarity is a boon and a bane. The fact that we are able to understand this phrase, even when we know that this is a wrong representation of the phrase “ খেলা হবে" , gives us an example of human cognitive adaptability. The downside however, is that we are now facing a problem of non-standard written representation of words and sentences, which might be a source material for quick jokes and laughter, but is a disaster for people who are enthusiastic about Bangla NLP. Often known as “Murad Takla”, this ever evolving form has created a unique problem for NLP researchers.


A form where words are written phonetically with English letters may be defined as “Romanized”. Example: The Bangla phrase “ খেলা হবে"  can be represented as “Khela Hobe”.

Romanized Bangla, often known as “Banglish” , is quite popular due to ease of use of english letters . “Murad Takla” seems to be  a subset of this “Banglish” representation.


However, a distinction is to be made. The “Pure Banglish” form “Khela Hobe” may not raise any eyebrows (specially because of acceptance as “Romanized Bangla” in recent NLP models such as : Roberta or even in the use of google translate) , the  “Murad Takla” form “Kala Haba” may create headaches. A somewhat understandable way to distinguish these two is to say that,

“Pure Banglish”= phonetically correct romanization

And “Murad Takla”=  phonetically in-correct romanization. Or “Mis-spelled Romanized Bangla”.

This oversimplified view may describe the “what”, but the question we need to dig is “how”.


How is “Murad Takla” formed?  It does seem quite random when people use it in social media but is there a rhyme or reason behind this madness?


Let’s take a closer look,

“Murad Takla” actually originates from the bangla phrase “মুরোদ থাকলে"

So how did “Murod Thakle” become “Murad Takla”?

Literally how?

Murod  ->  Murad  

vowel “o” after the letter “r”  was changed to “a” -- vowel change
Thakle  -> Takla  

Th has converted to T or in other words the “h” has been reduced 
 Vowel  “e” after the letter “l” was changed to “a” -- vowel change



Again, let’s consider something else,

Khela - > Kala

Kh has converted to K or in other words the “h” has been reduced 
 Vowel  “e” after the letter “h”  was changed to “a” -- vowel change
Hobe -> Haba

Vowel  “o” after the letter “H”  was changed to “a” -- vowel change
Vowel  “e” after the letter “b”  was changed to “a” -- vowel change

But does this rule of reducing a trailing “h” and random changing of vowels apply all across the board? Well ,not exactly.

Let’s consider another common word and some of its variances, "ভাই" Romanized- Bhai may come in the form of :

Bhae - vowel change (VC)
Bhaai/Bahai- vowel Addition (VA)
Bhi - vowel Reduction (VR)
Bai- Trailing H reduction (THR)
Bi- A combination of THR and VC and VR
Will also come in the form of : Vai or Vae or Vy and so on . Now here we are introduced with the letter “V” and although previously only vowels were replacing vowels , we see a semi-vowel (y) replacing a vowel. For simplicity , we will consider the addition/change/reduction of “W” and “Y” i.e- semi-vowels within VA/ VC/ VR.

A general way to put this ,could be “Replacement of Pseudo Similar Sounding letters”(RPSS)

Example of RPSS: (<-> means interchangeable )

“Bh” or “B” <-> “V”  
“Ph” or “P”<->”F”
“Ch”or “C”<->”S”
 “G” or ”J”<->”Z”
”C”<->”K”
and so on.

To get ourselves more comfortable with these operations (VC,VR,VA,THR,RPSS) lets try to figure out the following “Murad Takla” sentence:

“Gebar caba care j gan ”

So - “Jiber Sheba Kore je jon”-

Here :

Jiber to Gebar - has -

J->G:RPSS
i->e:VC
e->a:VC
Similarly-

sheba - caba - has

 Sh-> C: THR,RPSS
 e-> a : VC

Tryout the rest on your own . Or even better, come up with your own “Murad Takla” sentences and see which operations are happening underneath.  

A caution to keep in mind that these operations, while applicable to most, will not cover the whole spectrum of “Murad Takla”.

reference : https://www.markopolo.ai/blog/articles/kala-hoba by @mnansary
