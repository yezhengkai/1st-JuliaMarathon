import Images


function get_imgs_size_info(imgs_path)
    # https://stackoverflow.com/questions/43188201/reading-image-info-in-julia-without-loading-the-images
    wand = MagickWand()
    imgs_size = Tuple[]
    for img_path in imgs_path
        ImageMagick.pingimage(wand, img_path)
        push!(imgs_size, size(wand))  # (width, height)
    end
    info = DataFrame(path=imgs_path, size=imgs_size);
    return info
end

function resize_save_imgs(info_train_raw, train_raw_imgs_path, valid_raw_imgs_path; train_img_width=150)
    
    train_img_width = 150;
    train_img_height = floor(Int, train_img_width * mean(last.(info_train_raw.size)) / mean(first.(info_train_raw.size)));
    train_img_size = (train_img_width, train_img_height)
    
    processed_train_imgs = imresize.(load.(train_raw_imgs_path), reverse(train_img_size)...);
    train_imgs_path = replace.(train_raw_imgs_path, r"raw" => s"processed");
    @. mkpath(dirname(train_imgs_path));
    Images.save.(train_imgs_path, processed_train_imgs);

    processed_valid_imgs = imresize.(load.(valid_raw_imgs_path), reverse(train_img_size)...);
    valid_imgs_path = replace.(valid_raw_imgs_path, r"raw" => s"processed");
    @. mkpath(dirname(valid_imgs_path));
    Images.save.(valid_imgs_path, processed_valid_imgs);
end