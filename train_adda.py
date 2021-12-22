import plot_tsne
import calculate_confusion_matrix


def make_variable(tensor, volatile=False):
  ''' function to make tensor variable '''
  if use_cuda:
    tensor = tensor.cuda()
  return Variable(tensor, volatile=volatile)


def train_adda(num_epochs,  # number of epochs to train
               lr,          # learning rate
               save_step,   # save model checkpoint every this number of epochs 
               encoder,        
               classifier,            
               discriminator, 
               src_data_loaders,      # dict of train/valid/test source domain dataloaders
               tgt_data_loader,       # target domain dataloader for training
               tgt_data_loader_small, # target domain dataloader for testing (smaller)
               combined_dataloader,   # source + target domain dataloader for plotting t-SNE
               src_test_dir,   # file directory containing source doamin test images
               tgt_test_dir,   # file directory containing source doamin test images
               alpha_CLS=1.,      # coefficient for class classification loss in computing the total loss for encoder
               alpha_DA=1.,       # coefficient for domain adaptation (confusion) loss in computing the total loss for encoder
               multi_label=False, # True for multi-label data, False for single-label data
               test_threshold=0.3 # output threshold for considering the object as present for multi-label data
               ):
  ''' ADDA Training using symmetric mapping '''
  ### Define loss for class-classification ###
  criterion_cls = nn.MSELoss() if multi_label else nn.CrossEntropyLoss() # MSE loss for soft-label 

  ### Define loss for domain-classification ###
  criterion_DA  = nn.CrossEntropyLoss()

  ### Define optimizers for encoder, classifier, and discriminator ###
  optimizer_encoder       = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.9))
  optimizer_classifier    = optim.Adam(classifier.parameters(), lr=lr, betas=(0.5, 0.9))
  optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

  len_data_loader  = min(len(src_data_loaders['train']), len(tgt_data_loader)) # len(tgt_unlabeled_data_loader))
  valid_loss_min = np.Inf
  prev_save = ""

  ### Move the models to GPU ###
  if use_cuda:
    discriminator.cuda()
    encoder.cuda()
    classifier.cuda()

  classification_losses_E, domain_confusion_losses_E, losses_D, accs_D = [], [], [], []
  train_loss_E_class, train_loss_E_domain, train_loss_D, train_acc_D = 0., 0., 0., 0.
  valid_loss_class, val_n_corr_class = 0., 0 # only check for class-classification for validation (no domain-related tasks)

  ### Start Training! ###
  for epoch in range(num_epochs):

    #### 1. Plot t-SNE plot with source and target domain features together ####
    plot_tsne(combined_dataloader, encoder)

    ### Start timing ###
    start = time.time()       


    ####################  2. Loop through Training batches  ####################
    for step, ((images_src, tgt_src), (images_tgt, _)) in enumerate(zip(src_data_loaders['train'], tgt_data_loader)): 
      ##########################################################################
      #######  2.1 Train Source Encoder & Classifier with class labels  ########
      ##########################################################################
      encoder.train()
      classifier.train()
      images_src, images_tgt = make_variable(images_src), make_variable(images_tgt)
      tgt_src = tgt_src.type(torch.FloatTensor).cuda() if multi_label else make_variable(tgt_src)
      optimizer_encoder.zero_grad()
      optimizer_classifier.zero_grad()

      ### Forward only SOURCE DOMAIN images through Encoder & Classifier ###
      output = encoder(images_src)    # [batch_size, n_classes]  (target: [batch_size])
      output = torch.flatten(output, 1)
      output = classifier(output)

      ### Calculate class-classification loss for Encoder and Classifier ###
      loss_CLS = criterion_cls(output, tgt_src)
      train_loss_E_class += loss_CLS.item() 


      ##########################################################################   
      #############  2.2 Train Discriminator with domain labels  ###############
      ##########################################################################
      discriminator.train()
      optimizer_discriminator.zero_grad()

      ### Forward pass through Encoder ###
      feat_src = encoder(images_src) 
      feat_tgt = encoder(images_tgt)
      
      ### Concatenate source domain and target domain features ###
      feat_concat = torch.cat((feat_src, feat_tgt), 0) # [batch_size*2, 2048, 1, 1]
      feat_concat = feat_concat.squeeze(-1).squeeze(-1)  # [batch_size*2, 2048]

      ### Forward concatenated features through Discriminator ###
      pred_concat = discriminator(feat_concat.detach())

      ### prepare source domain labels (1) and target domain labels (0) ###
      label_src = make_variable(torch.ones(feat_src.size(0)).long()) 
      label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
      label_concat = torch.cat((label_src, label_tgt), 0)

      ### Calculate domain-classification loss for Discriminator ###
      loss_discriminator = criterion_DA(pred_concat.squeeze(1), label_concat)

      ### Backward Propagation for Discriminator ###
      loss_discriminator.backward()
      optimizer_discriminator.step()

      ### Update running losses/accuracies ###
      train_loss_D += loss_discriminator.item()
      
      pred_cls = torch.squeeze(pred_concat.max(1)[1])
      train_acc_D += (pred_cls == label_concat).float().mean()


      ##########################################################################
      ############  2.3 Train Source Encoder w/ FAKE domain label  #############
      ##########################################################################
      ### Forward only TARGET DOMAIN images through Encoder ###
      feat_tgt = encoder(images_tgt)

      ### Forward only TARGET DOMAIN features through Discriminator ###
      pred_tgt = discriminator(feat_tgt.squeeze(-1).squeeze(-1))     
      label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long()) # prepare fake labels
      
      ### Calculate FAKE domain-classification loss for Encoder ###
      loss_DA = criterion_DA(pred_tgt.squeeze(1), label_tgt)
      train_loss_E_domain += loss_DA.item()

      ### For encoder and Classifier, 
      ### optimize class-classification & fake domain-classification losses together ###
      loss_total = alpha_CLS * loss_CLS +  alpha_DA * loss_DA
      loss_total.backward()
      optimizer_encoder.step()
      optimizer_classifier.step()


    #################### 3. Loop through Validation batches ####################
    encoder.eval()
    classifier.eval()
    for data, target in src_data_loaders['valid']:
      data = make_variable(data)
      target = target.type(torch.FloatTensor).cuda() if multi_label else make_variable(target)
      with torch.no_grad():
        output = encoder(data)    # [batch_size, n_classes]  (target: [batch_size])
        output = torch.flatten(output, 1)
        output = classifier(output)

      loss = criterion_cls(output, target)
      valid_loss_class += loss.item()
      if multi_label==False:
        output = output.cpu().detach().numpy()
        val_n_corr_class += int(sum([np.argmax(pred)==target[i] for i, pred in enumerate(output)]))


    ####################  4. Log train/validation losses  ######################
    train_acc_D = train_acc_D/min(len(src_data_loaders['train']), len(tgt_data_loader))
    print('\n-----Epoch: %d/%d-----'%(epoch+1, num_epochs))
    print('Train Classification Loss (E,C): %.3f  Train Domain Confusion Loss (E): %.3f  Valid Classification Loss (E,C): %.3f'%(train_loss_E_class, train_loss_E_domain, valid_loss_class))  
    print('Domain Classification Loss (D): %.3f  Domain Classification Accuracy (D): %.3f  elapsed time: %.1fs'%(train_loss_D, train_acc_D, time.time()-start))  
    if multi_label==False:
      valid_acc = val_n_corr_class/len(src_data_loaders['valid'].dataset)

    ### Reset running losses/accuracies to zero ###
    classification_losses_E.append(train_loss_E_class)
    domain_confusion_losses_E.append(train_loss_E_domain)
    losses_D.append(train_loss_D)
    accs_D.append(train_acc_D)
    train_loss_E_class, train_loss_E_domain, train_loss_D, running_acc_D, val_n_corr = 0., 0., 0., 0., 0

        
    #########  5. Show confusion matrices for both domains' test sets  #########
    # set threshold=0.5 for source domain confusion matrix 
    cm = calculate_confusion_matrix(encoder, classifier, transform=transform_, classes=src_data_loaders['train'].dataset.classes, 
                                    img_dir=src_test_dir, threshold=0.5, multi_label=False)#, test=True)
    print("--Source Domain Confusion Matrix--")
    print(cm)
    # to be more lenient for target domain class deteciton, set threshold to be lower than 0.5 (e.g. 0.2)
    cm = calculate_confusion_matrix(encoder, classifier, transform=transform_, classes=tgt_data_loader_small.dataset.classes, 
                                    img_dir=tgt_test_dir, threshold=test_threshold, multi_label=True)#, test=True)
    print("--Target Domain Confusion Matrix--")
    print(cm)
    print()


    ######################  6. Save model checkpoints  #########################
    ### Save model if validtion loss is smaller than previous epoch's ###
    if valid_loss_class < valid_loss_min:
      ### Delete previously saved model checkpoint ###
      if prev_save:
        os.remove("encoder" + prev_save + ".pt")
        os.remove("classifier" + prev_save + ".pt")
      prev_save = "_" + str(epoch+1) 

      ### Save the new (best) model checkpoints ###
      torch.save(encoder.state_dict(), "encoder" + prev_save + ".pt")
      torch.save(classifier.state_dict(), "classifier" + prev_save + ".pt")
      valid_loss_min = valid_loss_class

    ### Regularly save model checkpoints every [save_step] epochs ###
    if ((epoch + 1) % save_step == 0):
      torch.save(encoder.state_dict(), "ADDA-encoder-{}.pt".format(epoch + 1))
      torch.save(classifier.state_dict(), "ADDA-classifier-{}.pt".format(epoch + 1))

  return encoder, classifier, classification_losses_E, domain_confusion_losses_E, losses_D, accs_D
