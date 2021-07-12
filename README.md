## Voice-Converter
Final project of CCBDA-2019-Spring-0760222

# Idea
In the cartoon “Detective Canon”, we can always see Conan hide in the Dark and use the voice converter to mimic the voice of the Detective Maori Kogoro to solve the homicide cases. 
We aim to create such an application that enables everyone to change the voice to another person’s tone.
# Technique
 
a.	one to one conversion - CycleGAN  
  We can translate one person’s sound into another person restricted by the source and target training data.  
  
b.	many to many conversion - StarGAN (Optional)  
  Extension of CycleGAN which is able to simultaneously learn many-to-many mappings across different attribute domains using a single generator network 
  
c.	Using non-parallel data  
  While training, we aim at using non-parallel data, which means we don't need two persons speaking the same set of sentences. Parallel data are only used to evaluate our model performance.  

# Uniqueness or the comparisons with state-of-the-art
a.	Add our favorite actor’s voice or our voice into target data set  

b.	(Optional) Combine the model into a real-time system, such that when a person speaks into the microphone, the speaker instantly outputs the target person's voice  
# Dataset
a.	Our sound include two males and two females

b.	Voice Conversion Challenge (VCC) 2018 dataset  

c.	(Optional) Our voice, teacher’s voice, celebrities’ voice (Detective Conan, weekly addresses by Trump)  

# System Architecture
![image](https://github.com/jasonyl13579/Voice-Converter/blob/main/picture/structure.png)
* RNN  
![image](https://github.com/jasonyl13579/Voice-Converter/blob/main/picture/rnn.png)
* GMM  
![image](https://github.com/jasonyl13579/Voice-Converter/blob/main/picture/gmm.png)
* StarGAN  
![image](https://github.com/jasonyl13579/Voice-Converter/blob/main/picture/stargan.png)
* IOS APP  
![image](https://github.com/jasonyl13579/Voice-Converter/blob/main/picture/ios.jpg)
# Reference
[1] Parallel-Data-Free Voice Conversion Using Cycle-Consistent   Adversarial Networks - https://arxiv.org/abs/1711.11293  

[2] StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks - https://arxiv.org/pdf/1806.02169.pdf  

[3] CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion - https://arxiv.org/abs/1904.04631  
