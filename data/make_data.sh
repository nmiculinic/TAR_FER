for ff in sts2017.gs.zip sts2017.eval.v1.1.zip; do
	if [[ ! -f $ff ]]; then
		curl http://alt.qcri.org/semeval2017/task1/data/uploads/$ff -o $ff
		unzip -f $ff
		echo "processed $ff"
	fi
done	

cp --update STS2017.eval.v1.1/STS.input.track5.en-en.txt train-en-en.in
cp --update STS2017.gs/STS.gs.track5.en-en.txt train-en-en.out
