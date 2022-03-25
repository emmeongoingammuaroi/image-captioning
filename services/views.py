import base64
import os
import random
import string

import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from nltk.translate.bleu_score import sentence_bleu

from .attention import evaluate_attention, img_name_val, cap_val, index_word, img_to_cap
from .transformer import evaluate_transformer


def index(request):
	return render(request, 'index.html')


@csrf_exempt
@require_POST
def evaluate_caption(request):
	image = request.FILES.get('image')
	model = request.POST.get('model')
	if image:
		image_path = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
		image_path = "./temp/" + image_path + ".jpg"
		with open(image_path, "wb") as f:
			f.write(image.read())
	else:
		image_path = random.choice(list(set(img_name_val)))

	if model == 'CNN-Attention-GRU':
		caption = evaluate_attention(image_path)[0]
		if caption:
			for i in caption:
				if i == "<unk>":
				    caption.remove(i)
		caption = ' '.join(caption)
		predict_caption = caption.rsplit(' ', 1)[0]
	elif model == 'CNN-Transformer':
		caption = evaluate_transformer(image_path)[0]
		if caption:
			for i in caption:
				if i == "<unk>":
				    caption.remove(i)
		predict_caption = ' '.join(caption)

	result = ''
	if image:
		result = f'<p>Predict caption: <span class="text-danger">{predict_caption}</span></p>'
		data = {
		    'text': result
		}
	else:
		real_caps = img_to_cap[image_path]
		reference = []
		real_captions = []
		for cap in real_caps:
			real_caption = ' '.join(cap.split()[1:-1])
			real_captions.append(real_caption)
			reference.append(real_caption.split())

		candidate = predict_caption.split()
		bleu1_score = sentence_bleu(reference, candidate, weights=(1.0, 0, 0, 0))
		bleu2_score = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
		bleu3_score = sentence_bleu(reference, candidate, weights=(0.3, 0.3, 0.3, 0))
		bleu4_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

		result = 'Real captions:<ul>'

		for i in range(len(real_captions)):
			result += f'<li class="text-danger">{real_captions[i]}</li>'

		result += '</ul>'
		result += f'<p>Predict caption: <span class="text-danger">{predict_caption}</span></p>' + \
	        f'<p>BLEU-1 score: <span class="text-danger">{bleu1_score*100}</span></p>' + \
	        f'<p>BLEU-2 score: <span class="text-danger">{bleu2_score*100}</span></p>' + \
	        f'<p>BLEU-3 score: <span class="text-danger">{bleu3_score*100}</span></p>' + \
	        f'<p>BLEU-4 score: <span class="text-danger">{bleu4_score*100}</span></p>'
		data = {
		    'text': result,
		    'image': 'http://127.0.0.1:8000' + image_path[1:]
		}
	return JsonResponse(data) 
