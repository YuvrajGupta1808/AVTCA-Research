# AVTCA Progress — Current Verified Results
**RAVDESS · 8-Class Emotion Recognition · 2026-05-28**

## Current Run Results

<table>
  <thead>
    <tr>
      <th>Run</th>
      <th>Best Val Top-1</th>
      <th>Test Top-1</th>
      <th>Test Top-5</th>
      <th>F1-Weighted</th>
      <th>F1-Micro</th>
      <th>Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>baseline_mel_h4</td><td>82.9167% (ep 40)</td><td>81.8750%</td><td>99.7917%</td><td>81.7620%</td><td>81.8750%</td><td>0.866828</td></tr>
    <tr><td>specaugment_mel_h4</td><td>80.0000% (ep 24)</td><td>79.3750%</td><td>99.5833%</td><td>79.4753%</td><td>79.3750%</td><td>0.912555</td></tr>
    <tr><td>audio_channel_gate_mel_h4</td><td>81.6667% (ep 6)</td><td>76.8750%</td><td>99.1667%</td><td>76.6528%</td><td>76.8750%</td><td>1.063809</td></tr>
    <tr><td>specaugment_audio_gate_mel_h4</td><td>81.6667% (ep 10)</td><td>68.9583%</td><td>100.0000%</td><td>68.9105%</td><td>68.9583%</td><td>1.103064</td></tr>
  </tbody>
</table>

## Run Changes

<table>
  <thead>
    <tr><th>Run</th><th>Recorded Change</th><th>Test Top-1</th></tr>
  </thead>
  <tbody>
    <tr><td>baseline_mel_h4</td><td>4 heads; LR 0.010; SpecAugment=false; audio_channel_attention=false</td><td>81.8750%</td></tr>
    <tr><td>specaugment_mel_h4</td><td>4 heads; LR 0.010; SpecAugment=true; audio_channel_attention=false</td><td>79.3750%</td></tr>
    <tr><td>audio_channel_gate_mel_h4</td><td>4 heads; LR 0.010; SpecAugment=false; audio_channel_attention=true</td><td>76.8750%</td></tr>
    <tr><td>specaugment_audio_gate_mel_h4</td><td>4 heads; LR 0.010; SpecAugment=true; audio_channel_attention=true</td><td>68.9583%</td></tr>
  </tbody>
</table>

## Per-Class Test Accuracy

<table>
  <thead>
    <tr><th>Emotion</th><th>baseline_mel_h4</th><th>specaugment_mel_h4</th><th>audio_channel_gate_mel_h4</th><th>specaugment_audio_gate_mel_h4</th></tr>
  </thead>
  <tbody>
    <tr><td>neutral</td><td>65.62%</td><td>75.00%</td><td>37.50%</td><td>75.00%</td></tr>
    <tr><td>calm</td><td>71.88%</td><td>71.88%</td><td>53.12%</td><td>45.31%</td></tr>
    <tr><td>happy</td><td>89.06%</td><td>81.25%</td><td>85.94%</td><td>57.81%</td></tr>
    <tr><td>sad</td><td>71.88%</td><td>64.06%</td><td>93.75%</td><td>56.25%</td></tr>
    <tr><td>angry</td><td>100.00%</td><td>100.00%</td><td>90.62%</td><td>62.50%</td></tr>
    <tr><td>fearful</td><td>68.75%</td><td>68.75%</td><td>56.25%</td><td>70.31%</td></tr>
    <tr><td>disgust</td><td>100.00%</td><td>100.00%</td><td>100.00%</td><td>100.00%</td></tr>
    <tr><td>surprised</td><td>79.69%</td><td>71.88%</td><td>78.12%</td><td>87.50%</td></tr>
  </tbody>
</table>

## Per-Class Test F1

<table>
  <thead>
    <tr><th>Emotion</th><th>baseline_mel_h4</th><th>specaugment_mel_h4</th><th>audio_channel_gate_mel_h4</th><th>specaugment_audio_gate_mel_h4</th></tr>
  </thead>
  <tbody>
    <tr><td>neutral</td><td>68.85%</td><td>66.67%</td><td>48.00%</td><td>57.83%</td></tr>
    <tr><td>calm</td><td>83.64%</td><td>83.64%</td><td>69.39%</td><td>62.37%</td></tr>
    <tr><td>happy</td><td>92.68%</td><td>87.39%</td><td>92.44%</td><td>72.55%</td></tr>
    <tr><td>sad</td><td>70.23%</td><td>62.12%</td><td>65.22%</td><td>52.17%</td></tr>
    <tr><td>angry</td><td>94.81%</td><td>92.09%</td><td>82.86%</td><td>76.92%</td></tr>
    <tr><td>fearful</td><td>71.54%</td><td>69.84%</td><td>70.59%</td><td>69.23%</td></tr>
    <tr><td>disgust</td><td>91.43%</td><td>84.77%</td><td>87.07%</td><td>79.50%</td></tr>
    <tr><td>surprised</td><td>74.45%</td><td>82.88%</td><td>83.33%</td><td>75.17%</td></tr>
  </tbody>
</table>

## Current Best

<table>
  <thead>
    <tr><th>Run</th><th>Best Val Top-1</th><th>Test Top-1</th><th>Test Top-5</th><th>UAR</th><th>F1-Weighted</th><th>F1-Macro</th></tr>
  </thead>
  <tbody>
    <tr><td>baseline_mel_h4</td><td>82.9167% (ep 40)</td><td>81.8750%</td><td>99.7917%</td><td>80.8594%</td><td>81.7620%</td><td>80.9552%</td></tr>
  </tbody>
</table>

## Historical Run History

<table>
  <thead>
    <tr>
      <th>Run</th>
      <th>Test Top-1</th>
      <th>Test Top-5</th>
      <th>UAR</th>
      <th>F1-Weighted</th>
      <th>F1-Macro</th>
      <th>Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>spec_01_baseline</td><td>66.6667%</td><td>96.6667%</td><td>66.9922%</td><td>65.6612%</td><td>65.8484%</td><td>1.312542</td></tr>
    <tr><td>spec_02_retrain_h4_e70</td><td>60.0000%</td><td>98.5417%</td><td>60.3516%</td><td>58.8741%</td><td>59.5695%</td><td>1.293564</td></tr>
    <tr><td>mel_h1_lr001_e75</td><td>70.8333%</td><td>98.3333%</td><td>70.8984%</td><td>70.6221%</td><td>70.4993%</td><td>1.063559</td></tr>
    <tr><td>mel_h8_lr001_e75</td><td>71.2500%</td><td>99.5833%</td><td>72.0703%</td><td>70.2509%</td><td>69.8308%</td><td>0.810862</td></tr>
    <tr><td>v2_h4_lr001_rerun1</td><td>75.2083%</td><td>99.5833%</td><td>72.8516%</td><td>74.8032%</td><td>73.2530%</td><td>1.010606</td></tr>
    <tr><td>v2_h8_lr001_rerun1</td><td>78.5417%</td><td>100.0000%</td><td>76.3672%</td><td>77.9270%</td><td>76.4879%</td><td>0.891395</td></tr>
    <tr><td>v2_h8_lr005_rerun1</td><td>77.5000%</td><td>98.7500%</td><td>76.7578%</td><td>77.1586%</td><td>77.1089%</td><td>0.993458</td></tr>
    <tr><td>v2_h8_e100_rerun1</td><td>72.9167%</td><td>100.0000%</td><td>69.7266%</td><td>72.1371%</td><td>69.8160%</td><td>1.036233</td></tr>
  </tbody>
</table>
