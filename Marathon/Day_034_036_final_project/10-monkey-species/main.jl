## References
# https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl
# https://github.com/FluxML/model-zoo/blob/master/vision/lenet_mnist/lenet_mnist.jl
# https://github.com/FluxML/model-zoo/blob/master/vision/dcgan_mnist/dcgan_mnist.jl

## import packages
using Flux
import Flux.Data: DataLoader
import Flux: @epochs, onecold, onehotbatch, throttle, logitcrossentropy
using Statistics
using CuArrays
using Printf

include("./src/monkey.jl")
import .monkey: read_label_discription, recursive_readdir, show_image, getarray


## setting
data_dir = joinpath("data");
train_data_dir = joinpath(data_dir, "training");
valid_data_dir = joinpath(data_dir, "validation");
label_txt = joinpath(data_dir, "monkey_labels.txt");
train_imgs_path = recursive_readdir(train_data_dir);
valid_imgs_path = recursive_readdir(valid_data_dir);
img_size = (100, 100);

# read monkey_labels.txt
label_discription = read_label_discription(label_txt);

# show ranfom image
idx = rand(1:lastindex(train_imgs_path));
show_image(train_imgs_path[idx]; return_plot=false);

# read training images and labels
tmp_imgs = getarray.(train_imgs_path; img_size=img_size);

num_img = length(tmp_imgs);
train_imgs = Array{Float32, 4}(undef, img_size..., 3, num_img);
for i in 1:num_img
    train_imgs[:, :, :, i] = convert(Array{Float32}, popfirst!(tmp_imgs));
end
train_labels = reduce(hcat, splitpath.(dirname.(train_imgs_path)))[end, :];
train_labels = onehotbatch(train_labels, unique(train_labels))

# read validation images and labels
tmp_imgs = getarray.(valid_imgs_path; img_size=img_size);
num_img = length(tmp_imgs);
valid_imgs = Array{Float32, 4}(undef, img_size..., 3, num_img);
for i in 1:num_img
    valid_imgs[:, :, :, i] = convert(Array{Float32}, popfirst!(tmp_imgs));
end

valid_labels = reduce(hcat, splitpath.(dirname.(valid_imgs_path)))[end, :];
valid_labels = onehotbatch(valid_labels, unique(valid_labels));


## hyper parameters
epochs = 30;
batchsize = 128;
learining_rate = 0.01;
decay = 0.1;
decay_step = 5;
clip = 1e-4;
optimizer = Flux.Optimiser(
    ExpDecay(learining_rate, decay, decay_step, clip),
    ADAM(learining_rate)
);

## build NN model
function conv_unit(chanel, nb_filters, mp=false)
    conv_bn = Chain(
        Conv((3, 3), chanel => nb_filters, leakyrelu, pad=1, stride=1),
        BatchNorm(nb_filters)
    )
    mp ? Chain(conv_bn..., x -> maxpool(x, (3, 3), stride=2, pad=1)) : conv_bn
end
function my_model()
    model = Chain(
        Conv((3, 3), 3=>32, pad=(1, 1), leakyrelu),
        MaxPool((2, 2)),
        Conv((3, 3), 32=>64, pad=(1, 1), leakyrelu),
        # MaxPool((2, 2)),
        Conv((3, 3), 64=>128, pad=(1, 1), leakyrelu),
        # MaxPool((2, 2)),
        Conv((3, 3), 128=>128, pad=(1, 1), leakyrelu),
        MaxPool((2, 2)),
        Conv((3, 3), 128=>64, pad=(1, 1), leakyrelu),
        # MaxPool((2, 2)),
        Conv((3, 3), 64=>32, pad=(1, 1), leakyrelu),
        # MaxPool((2, 2)),
        Conv((3, 3), 32=>16, pad=(1, 1), leakyrelu),
        flatten,
        Dense(10000, 512, leakyrelu),
        Dense(512, 128, leakyrelu),
        # Dense(128, 64, leakyrelu),
        Dense(128, 10),
        softmax
    )
end
model = my_model();

## loss function
# We augment `x` a little bit here, adding in random noise.
# augment(x) = x .+ gpu(0.01f0 * randn(eltype(x), size(x)))
# function loss(x, y)
#     x̂ = augment(x)
#     ŷ = model(x̂)
#     return logitcrossentropy(ŷ, y)
# end
loss(x, y) = logitcrossentropy(model(x), y);
# loss(x, y) = Flux.crossentropy(model(x), y);
# loss(ŷ, y) = logitcrossentropy(ŷ, y)

## callback function
# function eval_loss_accuracy(loader, model, device)
#     l = 0f0
#     acc = 0
#     ntot = 0
#     for (x, y) in loader
#         x, y = x |> device, y |> device
#         ŷ = model(x)
#         l += loss(ŷ, y) * size(x)[end]
#         acc += sum(onecold(ŷ |> device) .== onecold(y |> device))
#         ntot += size(x)[end]
#     end
#     return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
# end
##utility functions
round4(x) = round(x, digits=4)

function eval_loss_accuracy()
   l = 0f0
   acc = 0
   num_minibatch = length(valid_loader)
   for (x, y) in valid_loader
       l += loss(x, y)
       acc += accuracy(x, y)
   end
   return (loss = l/num_minibatch |> round4,
           acc = acc/num_minibatch*100 |> round4)
end
# function valid_loss()
#     l = 0f0
#     for (x, y) in valid
#         l += loss(x, y)
#     end
    # l/length(valid)
# end
valid_loss = typemax(Float32)
num_early_stop, early_stop = 0, 3
function evalcb()
    valid_loss_tmp, acc = @show(eval_loss_accuracy())
    @printf("Number of early stop: %i\n", num_early_stop)
    if valid_loss_tmp > valid_loss
        global num_early_stop += 1
    else
        global num_early_stop = 0
    end
    if num_early_stop > early_stop
        Flux.stop()
    end
    global valid_loss = valid_loss_tmp
end
# evalcb() = @show(valid_loss());


## evalution function
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

## use CUDA
model = model |> gpu;
train_imgs = train_imgs |> gpu;
train_labels = train_labels |> gpu;
valid_imgs = valid_imgs |> gpu;
valid_labels = valid_labels |> gpu;


## data DataLoader
train_loader = DataLoader(
    train_imgs, train_labels; batchsize=batchsize, shuffle=true
);
valid_loader = DataLoader(
    valid_imgs, valid_labels, batchsize=batchsize
);


## training
# device = gpu;
# num_params(model) = sum(length, Flux.params(model))
# function train()
#     # args = Args(; kws...)
#     # args.seed > 0 && Random.seed!(args.seed)
#     # use_cuda = args.cuda && CUDAapi.has_cuda_gpu()
#     # if use_cuda
#     #     device = gpu
#     #     @info "Training on GPU"
#     # else
#     #     device = cpu
#     #     @info "Training on CPU"
#     # end
#
#     ## DATA
#     # train_loader, test_loader = get_data(args)
#     # @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"
#
#     ## MODEL AND OPTIMIZER
#     # model = LeNet5() |> device
#     model = my_model() |> device;
#     # @info "LeNet5 model: $(num_params(model)) trainable params"
#     @info "model: $(num_params(model)) trainable params"
#
#     ps = Flux.params(model)
#
#     # opt = ADAM(args.η)
#     # if args.λ > 0
#     #     opt = Optimiser(opt, WeightDecay(args.λ))
#     # end
#
#     ## LOGGING UTILITIES
#     # if args.savepath == nothing
#     #     experiment_folder = savename("lenet", args, scientific=4,
#     #                 accesses=[:batchsize, :η, :seed, :λ]) # construct path from these fields
#     #     args.savepath = joinpath("runs", experiment_folder)
#     # end
#     # if args.tblogger
#     #     tblogger = TBLogger(args.savepath, tb_overwrite)
#     #     set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
#     #     @info "TensorBoard logging at \"$(args.savepath)\""
#     # end
#
#     function report(epoch)
#         train = eval_loss_accuracy(train_loader, model, device)
#         valid = eval_loss_accuracy(valid_loader, model, device)
#         println("Epoch: $epoch   Train: $(train)   Validation: $(valid)")
#
#         # if args.tblogger
#         #     set_step!(tblogger, epoch)
#         #     with_logger(tblogger) do
#         #         @info "train" loss=train.loss  acc=train.acc
#         #         @info "test"  loss=test.loss   acc=test.acc
#         #     end
#         # end
#     end
#
#     ## TRAINING
#     @info "Start Training"
#     report(0)
#     valid_loss, _ = eval_loss_accuracy(valid_loader, model, device)
#     num_early_stop, early_stop = 0, 5
#     for epoch in 1:epochs
#         # p = ProgressMeter.Progress(length(train_loader))
#
#         for (x, y) in train_loader
#             x, y = x |> device, y |> device
#             gs = Flux.gradient(ps) do
#                 ŷ = model(x)
#                 loss(ŷ, y)
#             end
#             Flux.Optimise.update!(optimizer, ps, gs)
#             # ProgressMeter.next!(p)   # comment out for no progress bar
#         end
#
#
#         valid_loss_tmp, _ = eval_loss_accuracy(valid_loader, model, device)
#         if valid_loss_tmp > valid_loss
#             num_early_stop += 1
#         end
#         if num_early_stop > early_stop
#             break;
#         end
#         report(epoch)
#         # epoch % args.infotime == 0 && report(epoch)
#         # if args.checktime > 0 && epoch % args.checktime == 0
#         #     !ispath(args.savepath) && mkpath(args.savepath)
#         #     modelpath = joinpath(args.savepath, "model.bson")
#         #     let model=cpu(model), args=struct2dict(args)
#         #         BSON.@save modelpath model epoch args
#         #     end
#         #     @info "Model saved in \"$(modelpath)\""
#         # end
#     end
# end

# train()
@epochs epochs Flux.train!(
    loss, params(model), train_loader, optimizer, cb=throttle(evalcb, 30)
)


# model evalution
@printf("Accuracy: %.2f%%\n", accuracy(valid_imgs, valid_labels) * 100)
