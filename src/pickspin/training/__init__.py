"""The offline pipeline that trains stage 2 of Pick's classifier.

labels builds complexity labels from the judge's results and splits them 80/20 with a fixed seed
(standard library only), finetune fine-tunes DistilBERT on them (the [train] extra), and evaluate
measures the accuracy of the keyword lists, DistilBERT and the hybrid classifier on the validation
split (the [classifier] extra). Import from the modules.
"""
