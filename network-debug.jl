ENV["MOCHA_USE_NATIVE_EXT"] = "true"

using Mocha
using HDF5

function _get_common_layers(n_classes=3)
	# AH: n_classes: Number of classes in order to correctly format the output layer
	# Return the main network architecture, minus the input layer and the analysis layer. 
	# Reference:
	# http://nbviewer.jupyter.org/github/pluskid/Mocha.jl/blob/master/examples/ijulia/ilsvrc12/imagenet-classifier.ipynb
	layers = [
	  ConvolutionLayer(name="conv1", n_filter=96, kernel=(11,11), stride=(4,4),
    	bottoms=[:data], tops=[:conv1])
	  PoolingLayer(name="pool1", tops=[:pool1], bottoms=[:conv1],
	      kernel=(3,3), stride=(2,2), pooling=Pooling.Max())
	  LRNLayer(name="norm1", tops=[:norm1], bottoms=[:pool1],
	      kernel=5, scale=0.1, power=0.75) #orig_ 0.0001
	  ConvolutionLayer(name="conv2", tops=[:conv2], bottoms=[:norm1],
	      kernel=(5,5), pad=(2,2), n_filter=256, n_group=2, neuron=Neurons.ReLU())
	  PoolingLayer(name="pool2", tops=[:pool2], bottoms=[:conv2],
	      kernel=(3,3), stride=(2,2), pooling=Pooling.Max())
	  LRNLayer(name="norm2", tops=[:norm2], bottoms=[:pool2],
	      kernel=5, scale=0.1, power=0.75)
	  ConvolutionLayer(name="conv3", tops=[:conv3], bottoms=[:norm2],
	      kernel=(3,3), pad=(1,1), n_filter=384, neuron=Neurons.ReLU())
	  ConvolutionLayer(name="conv4", tops=[:conv4], bottoms=[:conv3],
	      kernel=(3,3), pad=(1,1), n_filter=384, n_group=2, neuron=Neurons.ReLU())
	  ConvolutionLayer(name="conv5", tops=[:conv5], bottoms=[:conv4],
	      kernel=(3,3), pad=(1,1), n_filter=256, n_group=2, neuron=Neurons.ReLU())
	  PoolingLayer(name="pool5", tops=[:pool5], bottoms=[:conv5],
	      kernel=(3,3), stride=(2,2), pooling=Pooling.Max())
	  InnerProductLayer(name="fc6", tops=[:fc6], bottoms=[:pool5],
	      output_dim=4096, neuron=Neurons.Sigmoid())
	  InnerProductLayer(name="fc7", tops=[:fc7], bottoms=[:fc6],
	      output_dim=2048, neuron=Neurons.Sigmoid())
	  InnerProductLayer(name="fc8", tops=[:fc8], bottoms=[:fc7],
	      output_dim=n_classes)
	  #InnerProductLayer(name="fc8", output_dim=n_classes,
	  #  bottoms=[:fc7], tops=[:fc8])
	]
	return layers
end


function _training_setup(backend, name="train-data",
	source="data/oven-train.txt")

	data_layer  = AsyncHDF5DataLayer(name="in_data", source=source,
    	batch_size=64, shuffle=true)
 	loss_layer = SoftmaxLossLayer(name="loss", bottoms=[:fc8,:label])
	common_layers = _get_common_layers()
	net = Net("imgnet-train", backend, [data_layer, common_layers..., loss_layer])
	return net
end



function _validation_setup(backend, name="validation_data",
		source="data/oven-val.txt")
	data_layer_val = AsyncHDF5DataLayer(name="in_data", source=source,
		batch_size=64)
	acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:fc8, :label])
	common_layers = _get_common_layers()
    outdata_layer = HDF5OutputLayer(name="summary", filename="layer_summary2", 
  		force_overwrite=true, bottoms=[:data, :conv1, :conv3, :conv5, :fc6, :fc7])
	net = Net("validation-net", backend, [data_layer_val, common_layers...,
		outdata_layer, acc_layer])
	return net
end


function _test_setup(backend)
	img_x = 227
	img_y = 227
	channels = 3
	mem_data = MemoryDataLayer(name="data", tops=[:data], batch_size=1,
    data=Array[zeros(Float64, img_x, img_y, channels, 1)])
    common_layers = _get_common_layers()
    outdata_layer = HDF5OutputLayer(name="summary", filename="layer_summary2", 
	  	force_overwrite=true, bottoms=[:fc6, :fc7])
	softmax_layer = SoftmaxLayer(name="prob", tops=[:prob], bottoms=[:fc8])
	net = Net("imagenet", backend, [mem_data, common_layers..., outdata_layer,
	 softmax_layer])
	return net
end

function _solver_setup(test_net)
	exp_dir = "snapshots-1"
	method = SGD()
	#max_iter=10000
	params = make_solver_parameters(method, max_iter=1000, regu_coef=0.0005,
	    mom_policy=MomPolicy.Fixed(0.9),
	    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
	    load_from=exp_dir)
	solver = Solver(method, params)

	setup_coffee_lounge(solver, save_into="$exp_dir/statistics.hdf5", every_n_iter=500)
	add_coffee_break(solver, TrainingSummary(), every_n_iter=100)
	add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=100)
	add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=500)
	return solver
end

function run_training(backend="cpu")

	if backend == "cpu"
		backend = CPUBackend()
	elseif backend == "gpu"
		backend = GPUBackend()
	end
	init(backend)

	train_net = _training_setup(backend)
	println(train_net)

	val_net = _validation_setup(backend)
	println(val_net)
	solver = _solver_setup(val_net)
	solve(solver, train_net)
	destroy(train_net)
	destroy(val_net)
	shutdown(backend)
end

function run_prediction(backend="cpu")

	if backend == "cpu"
		backend = CPUBackend()
	elseif backend == "gpu"
		backend = GPUBackend()
	end
	init(backend)

	prediction_net = _test_setup(backend)
	load_snapshot(prediction_net, "snapshots/snapshot-00100.jld")

	h5open("data/oven-train.hdf5") do f
		i = 1
		img_data = f["data"]
		width, height, channels, n_imgs = size(img_data)
		for i = 1:n_imgs
			println(i)
	    	get_layer(prediction_net, "data").data[1][:,:,channels,1] = img_data[:,:,channels,i]

			forward(prediction_net)
			println()
			println("Label probability vector:")
			println(prediction_net.output_blobs[:prob].data)
		end
	end


end

run_training()
#run_prediction()
