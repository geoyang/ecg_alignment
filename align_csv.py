import argparse
import numpy as np
import matplotlib.pyplot as plt

everbeat_csv_file_name_ending = '.eb.csv'
reference_csv_file_name_ending = '.ref.csv'

class CsvAlign:
    def __init__(self, everbeat_filename, reference_filename):
        self.everbeat_csv_full_filename = everbeat_filename + everbeat_csv_file_name_ending
        self.reference_csv_full_filename = reference_filename + reference_csv_file_name_ending

    def read_csv(self, filename):
        return np.loadtxt(filename, dtype='int', usecols=0, delimiter=',', converters=float)

    def calc_r2(self, aligned_segment_length=4000):
        return np.corrcoef(self.everbeat_samples[:aligned_segment_length], self.reference_samples[:aligned_segment_length])

    def assess_quality(self, blanking_width=50, highpad=9000, lowpad=3000, eb_noise_truncation=0):
      dots = []
      for offset in range(highpad - lowpad):
          leftind = offset + eb_noise_truncation
          rightind = leftind + self.everbeat_samples.shape[0]
          rightind = min(rightind, self.reference_samples.shape[0])

          adjusted_everbeat_samples = self.everbeat_samples[:rightind - leftind]
          dotprod = np.dot(self.reference_samples[leftind:rightind], adjusted_everbeat_samples)
          dots.append(dotprod)

      self.dots = np.array(dots)

      if len(self.dots) == 0:
          raise ValueError("No valid dot products were calculated. Please check your input data.")

      absolute_max = np.max(self.dots)
      argmax = np.argmax(self.dots)

      leftpad = max(0, argmax - blanking_width)
      rightpad = min(len(self.dots), argmax + blanking_width)

      max_left = np.max(self.dots[:leftpad]) if leftpad > 0 else -np.inf
      max_right = np.max(self.dots[rightpad:]) if rightpad < len(self.dots) else -np.inf

      competitor = max(max_left, max_right)
      if competitor == -np.inf:
          competitor = 0

      q_fract = competitor / absolute_max

      print("Quality fraction is", q_fract)
      if q_fract < 0.9:
          print("\nThe alignment is likely successful.")
      else:
          print("\nSince the peak alignment was not very strongly unique, manual checking may be required.")

      return q_fract

    def plot_align(self, left_highlight=0, right_highlight=0, offset=-10):
        plt.figure(figsize=(20, 6))
        plt.plot(self.everbeat_samples, label='Everbeat Samples')
        plt.plot(self.reference_samples, label='Reference Samples', alpha=0.7)
        plt.legend()
        plt.show()
        if left_highlight and right_highlight:
            plt.axvspan(left_highlight, right_highlight, color='cyan', alpha=0.5)

    def align(self):
        self.everbeat_samples = self.read_csv(self.everbeat_csv_full_filename)
        self.reference_samples = self.read_csv(self.reference_csv_full_filename)
        print(f'Length of everbeat samples: {len(self.everbeat_samples)}')
        print(f'Length of reference samples: {len(self.reference_samples)}')
        self.assess_quality()
        self.plot_align()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align Everbeat and Reference CSV files.")
    parser.add_argument("eb_csv_filename", help="Everbeat CSV file", type=str)
    parser.add_argument("reference_csv_filename", help="Reference CSV file", type=str)
    args = parser.parse_args()

    csv_align = CsvAlign(args.eb_csv_filename, args.reference_csv_filename)
    csv_align.align()
