def scale_to_01_range(x):  
  ''' scale and move the coordinates so they fit [0; 1] range '''
  value_range = (np.max(x) - np.min(x))
  starts_from_zero = x - np.min(x)
  return starts_from_zero / value_range


def plot_tsne(dataloader, # contains data whose encoded features will be plotted
              encoder,    # model to pass data through to extract features 
              plot_imgs=False, # for each feature, plot images if True, plot coloured dots if False
              model_type='resnet50'): # supports one of 'resnet50' or 'vgg16'

  assert model_type in ['resnet50', 'vgg16'], 'model_type must be one of "resnet50" or "vgg16"!'
  encoder = encoder.cuda().eval()
  for i, (data, target, fname) in enumerate(dataloader):
    data = data.cuda() 
    with torch.no_grad():
      if model_type == "resnet50":
        outputs = encoder(data)
        outputs = torch.flatten(outputs, 1)
      elif model_type == "vgg16": 
        outputs = encoder.features(data)
        outputs = encoder.avgpool(outputs)
        outputs = torch.flatten(outputs, 1)
        outputs = encoder.classifier[0](outputs)
        outputs = encoder.classifier[1](outputs)
        outputs = encoder.classifier[2](outputs)
        outputs = encoder.classifier[3](outputs)
        outputs = encoder.classifier[4](outputs)
    outputs = outputs.cpu().numpy()
    features = outputs if i==0 else np.concatenate((features, outputs), axis=0)  
    labels = target if i==0 else np.concatenate((labels, target), axis=0)
    fnames = list(fname) if i==0 else fnames + list(fname) 

  print("# of samples : {} \n feature-dim : {}".format(features.shape[0], features.shape[1]))
  tsne = TSNE(n_components=2).fit_transform(features)

  # extract x and y coordinates representing the positions of the images on T-SNE plot
  fig = plt.figure(figsize=(8,5))
  tx = scale_to_01_range(tsne[:, 0])
  ty = scale_to_01_range(tsne[:, 1])
  
  classes = dataloader.dataset.classes 
  class2idx = {c:i for i, c in enumerate(classes)}
  # define list of colours for coloured dots
  colors = ['#00ffff', '#ff4000', '#ffbf00', '#0080ff', '#FF00FF', '#00ffff', '#008000', '#80ff00', '#8000ff', '#CCCCFF']
  colors_per_class = {label:colors[i] for i, label in enumerate(classes)}
  if plot_imgs:
    width, height = 4000, 3000
    max_dim = 100
    full_image = Image.new('RGBA', (width, height))
    img_paths = fnames

  for label in colors_per_class:
    indices = [i for i, l in enumerate(labels) if l == class2idx[label]]
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)
    
    if plot_imgs:
      current_img_paths = np.take(img_paths, indices)
      for img, x, y in zip(current_img_paths, current_tx, current_ty):
        tile = Image.open(img)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
    else:
      color = colors_per_class[label]  
      ax = fig.add_subplot(111)
      ax.scatter(current_tx, current_ty, c=color, label=label, alpha=0.5)

  if plot_imgs:
    plt.figure(figsize = (16,12))
    plt.imshow(full_image)
  else:
    ax.legend(loc='best')
    plt.show()
