MASK = 0.2
LRATE = 0.001
EPOCHS = 250
ITERS = 5
LAYERS = 5
DROPOUT = 0.5

all: hodge linegraph random identity unsigned
	montage -mode concatenate -tile 5x2 results/*scatter* results/*training* results.png

data:
	./process_data.py

hodge: data
	./main.py --mask=$(MASK) \
		  --learning_rate=$(LRATE) \
		  --epochs=$(EPOCHS) \
		  --iters_per_epoch=$(ITERS) \
		  --layers=$(LAYERS) \
		  --dropout=$(DROPOUT) \
		  --shift=hodge > results/hodge.txt

linegraph: data
	./main.py --mask=$(MASK) \
		  --learning_rate=$(LRATE) \
		  --epochs=$(EPOCHS) \
		  --iters_per_epoch=$(ITERS) \
		  --layers=$(LAYERS) \
		  --dropout=$(DROPOUT) \
		  --shift=linegraph > results/linegraph.txt

random: data
	./main.py --mask=$(MASK) \
		  --learning_rate=$(LRATE) \
		  --epochs=$(EPOCHS) \
		  --iters_per_epoch=$(ITERS) \
		  --layers=$(LAYERS) \
		  --dropout=$(DROPOUT) \
		  --shift=random > results/random.txt

identity: data
	./main.py --mask=$(MASK) \
		  --learning_rate=$(LRATE) \
		  --epochs=$(EPOCHS) \
		  --iters_per_epoch=$(ITERS) \
		  --layers=$(LAYERS) \
		  --dropout=$(DROPOUT) \
		  --shift=identity > results/identity.txt

unsigned: data
	./main.py --mask=$(MASK) \
		  --learning_rate=$(LRATE) \
		  --epochs=$(EPOCHS) \
		  --iters_per_epoch=$(ITERS) \
		  --layers=$(LAYERS) \
		  --dropout=$(DROPOUT) \
		  --shift=unsigned > results/unsigned.txt

clean:
	rm results/* results.png *.npy
