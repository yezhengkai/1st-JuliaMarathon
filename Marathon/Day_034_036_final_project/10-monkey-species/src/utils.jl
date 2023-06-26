## import packages
import DataFrames
import Images
import Plots



## read monkey_labels.txt
function read_label_discription(label_txt::String)::DataFrames.DataFrame
    
    # initialize
    column_names = [];
    content = [];

    # read text file
    for (i, line) in enumerate(eachline(label_txt))
        strings = strip.(split(line, ","));
        # println(strings, typeof(strings))
        if all(strings .== "")
            break
        end
        if i == 1
            append!(column_names, Symbol.(strings));
            continue
        end
        push!(content, String.(strings));
    end
    # convert 1d vector to 2d array (content has been transposed)
    content = reduce(hcat, content);

    # construct dataframe
    label_discription = DataFrames.DataFrame();
    for (i, name) in enumerate(column_names)
        if i > 3
            column = parse.(Int, content[i, :]);
            label_discription[!, name] =column;
        else
            label_discription[!, name] =content[i, :];
        end
    end

    return label_discription;
end

## show image
function show_image(path::String; return_plot=false)
    img = Images.load(path);
    p = Plots.plot(img)
    display(p);
    if return_plot
        return p
    else
        return nothing
    end
end

## convert image to array
# https://stackoverflow.com/questions/59344708/strange-and-uninformative-error-in-my-simple-julia-flux-dense-model
function getarray(path::String; img_size=(100, 100))::Array
    # img_size: (height, width)
    
    # code to get image based on the number and convert it to float
    file = Images.load(path)
    file = Images.imresize(file, img_size...)
    X = convert(Array{Float64}, Images.channelview(file))
    # X = permutedims(X, (2, 3, 1))  # (channel, height, width) to (height, width, channel)
    X = permutedims(X, (3, 2, 1))  # (channel, height, width) to (width, height, channel)
    
    individual_image_in_float = X
    return individual_image_in_float
end


## https://github.com/FluxML/Metalhead.jl/blob/sf/training/training/ImageNet/dataset.jl
function recursive_readdir(root::String)::Array
    ret = String[]
    for (r, dirs, files) in walkdir(root)
        for f in files
            push!(ret, relpath(joinpath(r, f)))
        end
    end
    return ret
end