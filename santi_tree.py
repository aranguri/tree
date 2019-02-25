import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

#This is the entire network forward from having accessed an answer
class qa2index(nn.Module):
    def __init__(self,encoding_size =,dropout_rate=0.2,num_questions,question_size, kb_length):
        super(qa2index, self).__init__()

        self.glove = torchtext.vocab.GloVe(name='6B', dim=100)
        self.question_encoder = nn.LSTM(encoding_size,hidden_size,bidirectional=True, bias=True)
        #after autoencoder trained contexts -> we can add a weight to this layer.
        self.kb = nn.Linear(encoding_size,kb_length,bias=False)

        #multiplicative att between question encoding and the
        self.qa_l1 = nn.linear()
        self.qa_l2 = nn.linear()
        self.qa_l3 = nn.linear()
        self.qa_start = nn.linear()
        self.qa_end = nn.linear()

    def forward(self,question):
        embeddings = self.glove(question)
    	question_lengths = [len(s) for s in question]

    	q_enc_hiddens = None

        pack_padded_Q = pack_padded_sequence(embeddings, question_lengths)
        packed_q_enc_hiddens, (last_q_hidden, last_q_cell) = self.question_encoder(pack_padded_Q)
        q_enc_hiddens, _ = pad_packed_sequence(packed_q_enc_hiddens)

        soft_decision = self.kb(q_enc_hiddens)
        decision = torch.functional.softmax(soft_decision) #log_softmax?
        knowledge_bit = torch.dot(self.kb.weights, decision)
        # now we have the decision we want to multiply it again by self.kb

        z1 = self.qa_l1(knowledge_bit)
        a1 = F.ReLU(z1)
        z2 = self.qa_l2(a1)
        a2 = F.ReLU(z2)
        z3 = self.qa_l3(a2)
        a3 = F.ReLU(z3)

        z4_start = self.qa_start(a3)
        start = torch.functional.softmax(z4_start)

        z4_end = self.qa_end(a3)
        end = torch.functional.softmax(z4_end)

        return start, end

model = qa2index(
    encoding_size=10,
    vocab=,
    dropout_rate=0.2,
    num_questions=,
    question_size=,
    kb_length=,
)

for i in range(epochs):
    for x, (correct_start, correct_end) in batch:
        start, end = model.forward(x)






# class santi_tree(nn.Module):
# 	def __init__(self,num_questions,question_size):
# 		#we've decided to ignore all for the moment
# 		# if num_questions % 2 != 0:
# 		# 	Raise('num questions need to be an even number')
# 		# # self.tree_height = np.log(num_questions) #to base 2.
# 		#self.kb = tensor(num_questions,question_size,self.tree_height)
# 		self.kb = torch.zeros([num_questions,question_size])


# 	def remake(self):

# 		pass

# 	def forward(self,x):
# 		left = 0
# 		right = 1
# 		for i in range(self.tree_height):
# 			sim_l = np.dot(self.kb[i][left],x)
# 			sim_r = np.dot(self.kb[i][right],x)
# 			if sim_r >= sim_l:
# 				go right?

# 		torch.dot(x,)
# 		np.matmul()
