require 'nn'
require 'torch'
require 'nngraph'
require 'optim'
require 'misc.netdef'
require 'cutorch'
require 'cunn'
require 'hdf5'
cjson=require('cjson') 
LSTM=require 'misc.LSTM'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_h5','data_img.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data_prepro.json','path to the json file containing additional info and vocab')

-- Model parameter settings
cmd:option('-learning_rate',3e-4,'learning rate for rmsprop')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 3000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-batch_size',500,'batch_size for each iterations')
cmd:option('-max_iters', 25000, 'max number of iterations to run for ')
cmd:option('-input_encoding_size', 200, 'he encoding size of each token in the vocabulary')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer',2,'number of the rnn layer')
cmd:option('-common_embedding_size', 1024, 'size of the common embedding vector')
cmd:option('-num_output', 2, 'number of output answers')
cmd:option('-img_norm', 1, 'normalize the image feature. 1 = normalize, 0 = not normalize')

--check point
cmd:option('-save_checkpoint_every', 600, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model/', 'folder to save checkpoints')

-- misc
cmd:option('-backend', 'cunn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')

opt = cmd:parse(arg)
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Setting the parameters
------------------------------------------------------------------------

local model_path = opt.checkpoint_path
local batch_size=opt.batch_size
local embedding_size_q=opt.input_encoding_size
local lstm_size_q=opt.rnn_size
local nlstm_layers_q=opt.rnn_layer
local nhimage=1432
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dummy_output_size=1
local decay_factor = 0.99997592083 -- 50000
paths.mkdir(model_path)

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_ques_h5)
local dataset = {}
local h5_file = hdf5.open(opt.input_ques_h5, 'r')

dataset['question'] = h5_file:read('/ques_train'):all()
dataset['lengths_q'] = h5_file:read('/ques_length_train'):all()
--dataset['img_list'] = h5_file:read('/img_pos_train'):all()
dataset['answers'] = h5_file:read('/answers'):all()
dataset['answers_comp'] = h5_file:read('/answers_comp'):all()
h5_file:close()

print('DataLoader loading h5 file: ', opt.input_img_h5)
local h5_file = hdf5.open(opt.input_img_h5, 'r')
dataset['fv_im_yes'] = h5_file:read('images_yes'):all()
dataset['fv_im_no'] = h5_file:read('images_no'):all()
h5_file:close()

dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])

-- Normalize the image feature
if opt.img_norm == 1 then
	local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im_yes'],dataset['fv_im_yes']),2)) 
	dataset['fv_im_yes']=torch.cdiv(dataset['fv_im_yes'],torch.repeatTensor(nm,1,1432)):float()

	local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im_no'],dataset['fv_im_no']),2)) 
	dataset['fv_im_no']=torch.cdiv(dataset['fv_im_no'],torch.repeatTensor(nm,1,1432)):float() 
end

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count

collectgarbage() 

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
print('Building the model...')

buffer_size_q=dataset['question']:size()[2]

--Network definitions
--VQA
--embedding: word-embedding
embedding_net_q=nn.Sequential()
				:add(nn.Linear(vocabulary_size_q,embedding_size_q))
				:add(nn.Dropout(0.5))
				:add(nn.Tanh())

--encoder: RNN body
encoder_net_q=LSTM.lstm_conventional(embedding_size_q,lstm_size_q,dummy_output_size,nlstm_layers_q,0.5)

--MULTIMODAL
--multimodal way of combining different spaces
multimodal_net=nn.Sequential()
				:add(netdef.AxB(2*lstm_size_q*nlstm_layers_q,nhimage,common_embedding_size,0.5))
				:add(nn.Dropout(0.5))
				:add(nn.Linear(common_embedding_size,noutput))

--- clone all three networks
embedding_net_q_2 = embedding_net_q:clone('weight', 'bias')
encoder_net_q_2 = encoder_net_q:clone('weight', 'bias')
multimodal_net_2 = multimodal_net:clone('weight', 'bias')


-- define a combined criterion
ce1 = nn.CrossEntropyCriterion()
ce2 = nn.CrossEntropyCriterion()
criterion = nn.ParallelCriterion():add(ce1, 0.5):add(ce2, 0.5)

--Optimization parameters
dummy_state_q=torch.Tensor(lstm_size_q*nlstm_layers_q*2):fill(0)
dummy_state_q_2=torch.Tensor(lstm_size_q*nlstm_layers_q*2):fill(0)
dummy_output_q=torch.Tensor(dummy_output_size):fill(0)
dummy_output_q_2=torch.Tensor(dummy_output_size):fill(0)

if opt.gpuid >= 0 then
	print('shipped data function to cuda...')
	embedding_net_q = embedding_net_q:cuda()
	embedding_net_q_2 = embedding_net_q:cuda()
	encoder_net_q = encoder_net_q:cuda()
	encoder_net_q_2 = encoder_net_q_2:cuda()
	multimodal_net = multimodal_net:cuda()
	multimodal_net_2 = multimodal_net_2:cuda()
	criterion = criterion:cuda()
	dummy_state_q = dummy_state_q:cuda()
	dummy_state_q_2 = dummy_state_q_2:cuda()
	dummy_output_q = dummy_output_q:cuda()
	dummy_output_q_2 = dummy_output_q_2:cuda()
end

--Processings
embedding_w_q,embedding_dw_q=embedding_net_q:getParameters() 
embedding_w_q:uniform(-0.08, 0.08) 

embedding_w_q_2, embedding_dw_q_2=embedding_net_q_2:getParameters()
-- do I need to do this?
---embedding_w_q_2:uniform(-0.08, 0.08)

encoder_w_q,encoder_dw_q=encoder_net_q:getParameters() 
encoder_w_q:uniform(-0.08, 0.08)

encoder_w_q_2, encoder_dw_q_2=encoder_net_q_2:getParameters()


multimodal_w,multimodal_dw=multimodal_net:getParameters() 
multimodal_w:uniform(-0.08, 0.08) 

multimodal_w_2,multimodal_dw_2=multimodal_net_2:getParameters()


sizes={encoder_w_q:size(1),embedding_w_q:size(1),multimodal_w:size(1)} 

-- optimization parameter
local optimize={} 
optimize.maxIter=opt.max_iters 
optimize.learningRate=opt.learning_rate
optimize.update_grad_per_n_batches=1 

optimize.winit=join_vector({encoder_w_q,embedding_w_q,multimodal_w}) 
optimize.winit_2=join_vector({encoder_w_q_2, embedding_w_q_2, multimodal_w_2})

------------------------------------------------------------------------
-- Next batch for train
------------------------------------------------------------------------
function dataset:next_batch()

	local qinds=torch.LongTensor(batch_size):fill(0) 
	local iminds=torch.LongTensor(batch_size):fill(0) 	
	
	local nqs=dataset['question']:size(1) 
	-- we use the last val_num data for validation (the data already randomlized when created)

	for i=1,batch_size do
		qinds[i]=torch.random(nqs) 
	 	iminds[i] = qinds[i]
	end


	local fv_sorted_q=sort_encoding_onehot_right_align(dataset['question']:index(1,qinds),dataset['lengths_q']:index(1,qinds),vocabulary_size_q) 
	local fv_im_yes=dataset['fv_im_yes']:index(1,iminds) 
	local fv_im_no=dataset['fv_im_no']:index(1, iminds)
	local labels=dataset['answers']:index(1,qinds) 
	local labels_comp = dataset['answers_comp']:index(1, qinds)
	
	-- ship to gpu
	if opt.gpuid >= 0 then
		fv_sorted_q[1]=fv_sorted_q[1]:cuda() 
		fv_sorted_q[3]=fv_sorted_q[3]:cuda() 
		fv_sorted_q[4]=fv_sorted_q[4]:cuda() 
		fv_im_yes = fv_im_yes:cuda()
		fv_im_no = fv_im_no:cuda()
		labels = labels:cuda()
		labels_comp = labels_comp:cuda()
	end

	return fv_sorted_q, fv_im_yes, fv_im_no, labels, labels_comp, batch_size 
end

------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------

-- duplicate the RNN
local encoder_net_buffer_q=dupe_rnn(encoder_net_q,buffer_size_q) 
local encoder_net_buffer_q_2=dupe_rnn(encoder_net_q_2,buffer_size_q)

count = 1
-- Objective function
function JdJ(x)
	local params=split_vector(x,sizes) 
	--load x to net parameters--
	if encoder_w_q~=params[1] then
		encoder_w_q:copy(params[1]) 
		encoder_w_q_2:copy(params[1])

		for i=1,buffer_size_q do
			encoder_net_buffer_q[2][i]:copy(params[1]) 
			encoder_net_buffer_q_2[2][i]:copy(params[1])
		end
	end
	if embedding_w_q~=params[2] then
		embedding_w_q:copy(params[2]) 
		embedding_w_q_2:copy(params[2])
	end
	if multimodal_w~=params[3] then
		multimodal_w:copy(params[3]) 
		multimodal_w_2:copy(params[3])
	end

	--clear gradients--
	for i=1,buffer_size_q do
		encoder_net_buffer_q[3][i]:zero()
		encoder_net_buffer_q_2[3][i]:zero()
	end
	embedding_dw_q:zero() 
	multimodal_dw:zero() 

	embedding_dw_q_2:zero()
	multimodal_dw_2:zero()

	--grab a batch--
	local fv_sorted_q,fv_im_yes, fv_im_no, labels, labels_comp, batch_size=dataset:next_batch()
	--require('mobdebug').listen() 
	local question_max_length=fv_sorted_q[2]:size(1) 	

	--embedding forward--
	local word_embedding_q=split_vector(embedding_net_q:forward(fv_sorted_q[1]),fv_sorted_q[2]) 

	--second embedding forward
	local word_embedding_q_2=split_vector(embedding_net_q_2:forward(fv_sorted_q[1]), fv_sorted_q[2])

	--encoder forward--
	local states_q,junk2=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state_q:fill(0),batch_size,1),word_embedding_q,fv_sorted_q[2]) 
	
	-- second encoder forward
	local states_q_2,junk23=rnn_forward(encoder_net_buffer_q_2,
	torch.repeatTensor(dummy_state_q_2:fill(0), batch_size, 1),
	word_embedding_q_2, fv_sorted_q[2])

	local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q[4])	
	
	--multimodal/criterion forward. Forward through both nets--
	local scores1=multimodal_net:forward({tv_q,fv_im_yes})
	local scores2=multimodal_net_2:forward({tv_q,fv_im_no})

	local f = criterion:forward({scores1, scores2},{labels, labels_comp})

	local grad = criterion:backward({scores1, scores2},{labels, labels_comp})
		
	local tmp1=multimodal_net:backward({tv_q:cuda(),fv_im_yes:cuda()},grad[1])
	local dtv_q=tmp1[1]:index(1,fv_sorted_q[3])

	local tmp2=multimodal_net_2:backward({tv_q:cuda(), fv_im_no:cuda()},grad[2])
	local dtv_q_2=tmp2[1]:index(1, fv_sorted_q[3])
	
	--encoder backward
	local junk4,dword_embedding_q=rnn_backward(encoder_net_buffer_q, dtv_q, dummy_output_q, states_q, word_embedding_q, fv_sorted_q[2]) 

	local junk43,dword_embedding_q_2=rnn_backward(encoder_net_buffer_q_2, dtv_q_2, dummy_output_q_2, states_q_2, word_embedding_q_2,fv_sorted_q[2])

	--embedding backward--
	dword_embedding_q=join_vector(dword_embedding_q)
	dword_embedding_q_2=join_vector(dword_embedding_q_2)
	embedding_net_q:backward(fv_sorted_q[1],dword_embedding_q) 
	
	embedding_net_q_2:backward(fv_sorted_q[1], dword_embedding_q_2)

	--summarize f and gradient

	local encoder_adw_q=encoder_dw_q:clone():zero()
	local encoder_adw_q_2=encoder_dw_q_2:clone():zero()
	for i=1,question_max_length do
		-- encoder_adw is going to zero here
		encoder_adw_q=encoder_adw_q+encoder_net_buffer_q[3][i] 
		encoder_adw_q_2=encoder_adw_q_2+encoder_net_buffer_q_2[3][i]
	end


	gradients_1=join_vector({encoder_adw_q,embedding_dw_q,multimodal_dw}) 
	gradients_2=join_vector({encoder_adw_q_2, embedding_dw_q_2, multimodal_dw_2})
	
	gradients = (gradients_1+gradients_2)/2
	
	gradients:clamp(-10,10) 
	if running_avg == nil then
		running_avg = f
	end
	running_avg=running_avg*0.95+f*0.05
	print(f)
	return f, gradients
end


----------------------------------------------------------------------------------------------
-- Training
----------------------------------------------------------------------------------------------
-- With current setting, the network seems never overfitting, so we just use all the data to train

local state={}
for iter = 1, opt.max_iters do
--for iter = 1, 21090 do
	print(iter)
	if iter%opt.save_checkpoint_every == 0 then
		paths.mkdir(model_path..'save')
		torch.save(string.format(model_path..'save/lstm_save_iter%d.t7',iter),
			{encoder_w_q=encoder_w_q,embedding_w_q=embedding_w_q,multimodal_w=multimodal_w}) 
	end
	if iter%100 == 0 then
		print('training loss: ' .. running_avg, 'on iter: ' .. iter .. '/' .. opt.max_iters)
	end
	optim.rmsprop(JdJ, optimize.winit, optimize, state)
	
	optimize.learningRate=optimize.learningRate*decay_factor 
	if iter%50 == 0 then -- change this to smaller value if out of the memory
		collectgarbage()
	end
end

-- Saving the final model
torch.save(string.format(model_path..'lstm.t7',i),
	{encoder_w_q=encoder_w_q,embedding_w_q=embedding_w_q,multimodal_w=multimodal_w}) 
