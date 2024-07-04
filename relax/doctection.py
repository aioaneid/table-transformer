import deepdoctection as dd
from matplotlib import pyplot as plt
import argparse
import os
import pathlib

def main(pdf_file_path):
    analyzer = dd.get_dd_analyzer(language="deu", ocr=True)  # instantiate the built-in analyzer similar to the Hugging Face space demo

    df = analyzer.analyze(path=pdf_file_path)  # setting up pipeline
    df.reset_state()                 # Trigger some initialization
    doc = iter(df)

    output_dir = "/home/dai/work/tmp/doctection/" + pathlib.PurePosixPath(pdf_file_path).stem
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    
    j = 0
    while True:
        try:
            page = next(doc)
        except StopIteration:
            break
        image = page.viz()

        plt.figure(figsize = (25,17))
        plt.axis('off')
        image_path = f"{output_dir}/{j}.png"
        print(image_path)
        plt.imsave(fname=image_path, arr=image)
        j += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Process tables')
    parser.add_argument('-ip', '--input_pdf', help='One pdf as input.')
    args = parser.parse_args()

    main(args.input_pdf)
    
