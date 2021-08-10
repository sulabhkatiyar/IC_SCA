import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from new_utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

cudnn.benchmark = True  

captions_dump=True

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def evaluate(loader, encoder, decoder, criterion, word_map, device):
    global captions_dump
    
    empty_hypo = 0
    references = list()
    hypotheses = list()

    beam_size = 1
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)
    image_names = list()

    print("Evaluating on validation set...")

    # For each image
    for i, (image, caps, caplens, allcaps, image_name) in enumerate(
            tqdm(loader, desc="EVALUATING WITHOUT Teacher Forcing")):

        k = beam_size

        image = image.to(device)  

        encoder_out = encoder(image)  
        encoder_dim = encoder_out.size(-1)

        encoder_out = encoder_out.view(1, encoder_dim)  

        encoder_out = encoder_out.expand(k, encoder_dim)  
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  
        seqs = k_prev_words  

        top_k_scores = torch.zeros(k, 1).to(device)  

        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        img = decoder.get_img_rep(encoder_out)
        h1, c1 = torch.zeros_like(img), torch.zeros_like(img)
        h2, c2 = decoder.init_hidden_state(encoder_out)

        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)

            h1, c1 = decoder.decode_step1(embeddings * img,(h1, c1))                
            h2, c2 = decoder.decode_step2(torch.cat([embeddings, h1], dim = 1), (h2, c2))    
                
            scores = decoder.fc(h2)  
            scores = F.log_softmax(scores, dim=1)

            scores = top_k_scores.expand_as(scores) + scores 

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True) 
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True) 

            prev_word_inds = top_k_words / vocab_size  
            next_word_inds = top_k_words % vocab_size  

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1) 

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  

            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h1, c1 = h1[prev_word_inds[incomplete_inds]], c1[prev_word_inds[incomplete_inds]]
            h2, c2 = h2[prev_word_inds[incomplete_inds]], c2[prev_word_inds[incomplete_inds]]
            img = img[prev_word_inds[incomplete_inds]]

            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > 50:
                break
            step += 1


        try:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        except:
            seq = []
            empty_hypo += 1

        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}], img_caps))  
        references.append(img_captions)

        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        image_names.append(image_name)

        assert len(references) == len(hypotheses) == len(image_names)

    bleu4 = corpus_bleu(references, hypotheses)
    bleu3 = corpus_bleu(references, hypotheses, (1.0/3.0,1.0/3.0,1.0/3.0,))
    bleu2 = corpus_bleu(references, hypotheses, (1.0/2.0,1.0/2.0,))
    bleu1 = corpus_bleu(references, hypotheses, (1.0/1.0,))

    print("The Validation set BLEU scores for model are {}.\n".format([bleu1,bleu2,bleu3,bleu4]))

    with open('val_run_logs.txt', 'a') as eval_run:
        eval_run.write("The Validation set BLEU scores, with beam-size {}, for model are {}.\n".format(beam_size, [bleu1,bleu2,bleu3,bleu4]))

    return bleu1,bleu2,bleu3,bleu4
