const { useState: useStateTW, useEffect: useEffectTW, useRef: useRefTW } = React;

const GEN_PARAMS = [
  { key: 'max_new_tokens', label: 'Max new tokens', type: 'number', default: 256, min: 16,  max: 4096, step: 16, help: 'Upper bound on generated tokens. Truncates long outputs. Raise if answers are cut off.' },
  { key: 'do_sample',      label: 'Sample (random)', type: 'boolean', default: false,                             help: 'Off → greedy (deterministic). On → uses temperature / top_p / top_k.' },
  { key: 'temperature',    label: 'Temperature',    type: 'range',  default: 0.7, min: 0,   max: 2,    step: 0.05, help: 'Higher = more creative. Only applies when Sample is on.' },
  { key: 'top_p',          label: 'Top-p',          type: 'range',  default: 0.95, min: 0,  max: 1,    step: 0.01, help: 'Nucleus sampling. Only applies when Sample is on.' },
  { key: 'top_k',          label: 'Top-k',          type: 'number', default: 50,   min: 0,  max: 200,  step: 1,    help: '0 disables top-k. Only applies when Sample is on.' },
];
const TASK_META = {

  'image-segmentation':  { nm: 'Segment',     input: 'image', output: 'masks',  icon: 'eye',     accent: 'oklch(70% 0.14 155)',
    guide: {
      summary: 'Pixel-level segmentation. Every pixel gets a class label from the model\'s vocabulary.',
      rows: [{ k: 'Image', v: 'JPG/PNG/WebP. Anything the model can see', req: true }],
    },
    params: [

      { key: 'oneformer_mode', label: 'OneFormer mode', type: 'select', default: 'semantic',
        visibleWhen: (mid) => /oneformer/i.test(mid || ''),
        help: 'OneFormer trains all three heads on one checkpoint. Pick semantic for one mask per class, instance for one mask per object, panoptic for both stuff and things.',
        options: [
          { value: 'semantic', label: 'Semantic' },
          { value: 'instance', label: 'Instance' },
          { value: 'panoptic', label: 'Panoptic' },
        ],
      },
      { key: 'overlay_alpha',    label: 'Overlay opacity',       type: 'range', default: 140, min: 50, max: 255, step: 5,   help: 'Alpha of the mask overlay composited onto the image (0-255).' },
      { key: 'legend_min_pct',   label: 'Legend min coverage %', type: 'range', default: 0.3, min: 0,  max: 5,   step: 0.1, help: 'Hide classes covering less than this percent of the image from the legend.' },
    ]},
  'mask-generation':     { nm: 'SAM Segment', input: 'image', output: 'masks',  icon: 'sparkle', accent: 'oklch(72% 0.15 200)',
    guide: {
      summary: 'Class-agnostic mask generation. Two modes: (1) auto: SAM samples a grid of points and outlines every distinct object. (2) point-prompted: click the image to pick specific objects.',
      rows: [
        { k: 'Image',  v: 'JPG/PNG/WebP',                                           req: true  },
        { k: 'Points', v: 'Optional. Click the image to include/exclude regions.',  req: false },
      ],
    },
    params: [
      { key: 'points_per_batch', label: 'Points per batch', type: 'number', default: 64,  min: 16, max: 256, step: 16, help: 'Auto mode only: higher = finer grid of prompt points, slower run.' },
      { key: 'max_masks',        label: 'Max masks',        type: 'number', default: 64,  min: 4,  max: 256, step: 4,  help: 'Auto mode only: cap on returned regions.' },
      { key: 'min_mask_pct',     label: 'Min mask %',       type: 'range',  default: 0.5, min: 0,  max: 10,  step: 0.1, help: 'Auto mode only: discard masks covering less than this percent of the image.' },
      { key: 'overlay_alpha',    label: 'Overlay opacity',  type: 'range',  default: 140, min: 50, max: 255, step: 5,  help: 'Alpha of the mask overlay (0-255).' },
    ]},
  'object-detection':    { nm: 'Detect',      input: 'image', output: 'boxes',  icon: 'target',  accent: 'oklch(70% 0.14 155)',
    guide: {
      summary: 'Draws bounding boxes around objects from the model\'s fixed class list.',
      rows: [{ k: 'Image', v: 'JPG/PNG/WebP', req: true }],
    },
    params: [
      { key: 'threshold', label: 'Score threshold', type: 'range', default: 0.5,  min: 0, max: 1, step: 0.01, help: 'Drop detections below this confidence. Lower = more boxes, more noise.' },
      { key: 'nms_iou',   label: 'NMS IoU',         type: 'range', default: 0.45, min: 0, max: 1, step: 0.05, help: 'IoU for non-max suppression. Higher keeps more overlapping boxes.' },
    ]},
  'image-classification':{ nm: 'Classify',    input: 'image', output: 'labels', icon: 'eye',     accent: 'oklch(70% 0.14 155)',
    guide: {
      summary: 'Closed-vocabulary classifier. Returns top-k labels from the model\'s training classes.',
      rows: [{ k: 'Image', v: 'JPG/PNG/WebP', req: true }],
    },
    params: [
      { key: 'top_k', label: 'Top-K labels', type: 'number', default: 10, min: 1, max: 50, step: 1, help: 'How many labels to return, ranked by score.' },
    ]},
  'image-to-text':       { nm: 'Describe',    input: 'image', output: 'text',   icon: 'chat',    accent: 'oklch(70% 0.12 230)',
    textSlot: {
      label: 'Prompt (optional)',
      placeholder: 'Ask a question about the image, or leave empty for a caption.',
      required: false,
      help: 'Empty → caption. With text → visual question answering.',
    },
    guide: {
      summary: 'BLIP / GIT / Pix2Struct-style captioner. Describes the image, or answers a question if you provide one. Also covers TrOCR for line-level OCR (printed and handwritten).',
      rows: [
        { k: 'Image',  v: 'JPG/PNG/WebP. Also works for cropped text lines (TrOCR)', req: true  },
        { k: 'Prompt', v: 'Optional question for VQA',                                 req: false },
      ],
    },
    params: [
      { key: 'max_new_tokens', label: 'Max new tokens', type: 'number',  default: 60,  min: 16, max: 512,  step: 16,  help: 'Upper bound on the caption / answer length.' },
      { key: 'do_sample',      label: 'Sample (random)', type: 'boolean', default: false,                             help: 'Off → greedy. On → sampled generation using temperature / top_p.' },
      { key: 'temperature',    label: 'Temperature',    type: 'range',   default: 0.7, min: 0,  max: 2,    step: 0.05, help: 'Only applies when Sample is on.' },
      { key: 'top_p',          label: 'Top-p',          type: 'range',   default: 0.95, min: 0, max: 1,    step: 0.01, help: 'Only applies when Sample is on.' },
    ]},

  'depth-estimation':    { nm: 'Estimate depth', input: 'image', output: 'image', icon: 'eye', accent: 'oklch(70% 0.14 200)',
    guide: {
      summary: 'Monocular depth estimation (DPT, MiDaS, ZoeDepth, Depth Anything v1/v2, Depth Pro). Returns a colorized depth map at the input resolution.',
      rows: [{ k: 'Image', v: 'JPG/PNG/WebP. Indoor or outdoor scenes work', req: true }],
    },
    params: [
      { key: 'invert', label: 'Invert (near = warm)',  type: 'boolean', default: false,                    help: 'Flip the colormap. Some models predict inverse depth. Toggle if near and far look swapped.' },
      { key: 'blend',  label: 'Blend with original',   type: 'range',   default: 0,    min: 0, max: 1, step: 0.05, help: 'Alpha-blend the colored depth back onto the source image so geometry stays visible.' },
    ]},

  'document-question-answering': { nm: 'Read document', input: 'image', output: 'text', icon: 'chat', accent: 'oklch(70% 0.12 65)',
    textSlot: {
      label: 'Question (optional)',
      placeholder: 'What is the invoice total? Leave empty to extract all text',
      required: false,
      help: 'Ask a question about the document, or leave empty to OCR the page.',
    },
    guide: {
      summary: 'Document AI: Donut, LayoutLMv3 and friends. Reads scanned pages, receipts, forms, and answers questions about them. For pure OCR (TrOCR-style line recognition), use the Caption tab instead.',
      rows: [
        { k: 'Image',    v: 'Scan / photo of a document, receipt, form, or page',         req: true },
        { k: 'Question', v: 'Optional. e.g. "Total amount?", "Who signed this?"',         req: false },
      ],
      example: 'What is the total amount on this receipt?',
    },
    params: [
      { key: 'top_k', label: 'Top-K answers', type: 'number', default: 1, min: 1, max: 10, step: 1, help: 'Return multiple candidate answers ranked by score (extractive models only).' },
    ]},

  'zero-shot-image-classification': { nm: 'Classify (zero-shot)', input: 'image', output: 'labels', icon: 'eye', accent: 'oklch(70% 0.14 155)',
    textSlot: {
      label: 'Candidate labels',
      placeholder: 'cat, dog, a photo of a car at night',
      required: true,
      help: 'Comma-separated captions. The model scores each against the image (CLIP / SigLIP / MetaCLIP).',
    },
    guide: {
      summary: 'CLIP-family model. Scores an image against candidate captions you supply.',
      rows: [
        { k: 'Image',            v: 'JPG/PNG/WebP',                                           req: true },
        { k: 'Candidate labels', v: 'Comma-separated, e.g. "cat, dog, a photo of a panda"',   req: true },
      ],
      example: 'cat, dog, a photo of a car at night, sunset over mountains',
    },
    params: []},
  'zero-shot-object-detection':     { nm: 'Detect (zero-shot)',   input: 'image', output: 'boxes',  icon: 'target', accent: 'oklch(70% 0.14 155)',
    textSlot: {
      label: 'Candidate labels',
      placeholder: 'car, person, license plate',
      required: true,
      help: 'Comma-separated object names. OWL / Grounding-DINO localizes any that appear.',
    },
    guide: {
      summary: 'Open-vocabulary detector. You say what to find, it draws boxes.',
      rows: [
        { k: 'Image',            v: 'JPG/PNG/WebP',                                       req: true },
        { k: 'Candidate labels', v: 'Comma-separated object names, e.g. "car, person"',   req: true },
      ],
      example: 'red car, blue car, license plate, person wearing a helmet',
    },
    params: [
      { key: 'threshold', label: 'Score threshold', type: 'range', default: 0.1,  min: 0, max: 1, step: 0.01, help: 'Drop detections below this confidence. Zero-shot models usually need a much lower threshold than closed-set detectors.' },
      { key: 'nms_iou',   label: 'NMS IoU',         type: 'range', default: 0.45, min: 0, max: 1, step: 0.05, help: 'IoU for non-max suppression. Higher keeps more overlapping boxes.' },
    ]},

  'image-text-to-text':  { nm: 'Ask a VLM', input: 'image', output: 'text', icon: 'chat', accent: 'oklch(70% 0.12 230)',
    textSlot: {
      label: 'Prompt (optional)',
      placeholder: "What's happening in this image?",
      required: false,
      help: 'Empty → caption. With text → ask the VLM a question about the image.',
    },
    guide: {
      summary: 'Vision-language model (Qwen-VL, LLaVA, Idefics, SmolVLM, Florence-2, …). Ask it about the image.',
      rows: [
        { k: 'Image',  v: 'JPG/PNG/WebP',                        req: true  },
        { k: 'Prompt', v: 'Natural-language question, optional', req: false },
      ],
    },
    params: [
      { key: 'max_new_tokens', label: 'Max new tokens', type: 'number',  default: 512, min: 32, max: 2048, step: 32,  help: 'Upper bound on generated tokens.' },
      { key: 'do_sample',      label: 'Sample (random)', type: 'boolean', default: false,                             help: 'Off → greedy. On → sampled generation.' },
      { key: 'temperature',    label: 'Temperature',    type: 'range',   default: 0.7, min: 0,  max: 2,    step: 0.05, help: 'Only applies when Sample is on.' },
      { key: 'top_p',          label: 'Top-p',          type: 'range',   default: 0.95, min: 0, max: 1,    step: 0.01, help: 'Only applies when Sample is on.' },



      { key: 'florence_task', label: 'Florence-2 task', type: 'select', default: '<CAPTION>',
        visibleWhen: (mid) => /florence-?2/i.test(mid || ''),
        help: 'Florence-2 routes behavior via task tokens. Some tokens (CAPTION_TO_PHRASE_GROUNDING, REFERRING_EXPRESSION_SEGMENTATION, OPEN_VOCABULARY_DETECTION) also use the Prompt field as the phrase/expression to ground.',
        options: [
          { value: '<CAPTION>',                            label: 'Caption (short)' },
          { value: '<DETAILED_CAPTION>',                   label: 'Caption (detailed)' },
          { value: '<MORE_DETAILED_CAPTION>',              label: 'Caption (more detailed)' },
          { value: '<OD>',                                 label: 'Object detection' },
          { value: '<DENSE_REGION_CAPTION>',               label: 'Dense region captions' },
          { value: '<REGION_PROPOSAL>',                    label: 'Region proposals' },
          { value: '<CAPTION_TO_PHRASE_GROUNDING>',        label: 'Phrase grounding (uses prompt)' },
          { value: '<REFERRING_EXPRESSION_SEGMENTATION>',  label: 'Referring segmentation (uses prompt)' },
          { value: '<OPEN_VOCABULARY_DETECTION>',          label: 'Open-vocab detection (uses prompt)' },
          { value: '<OCR>',                                label: 'OCR (plain text)' },
          { value: '<OCR_WITH_REGION>',                    label: 'OCR with regions' },
        ],
      },
    ]},

  'automatic-speech-recognition': { nm: 'Transcribe', input: 'audio', output: 'text',  icon: 'waveform', accent: 'oklch(70% 0.13 65)',
    guide: {
      summary: 'Speech-to-text (Whisper, Wav2Vec2, Parakeet, …).',
      rows: [{ k: 'Audio', v: 'WAV / MP3 / FLAC / OGG / M4A', req: true }],
    },
    params: [

      { key: 'whisper_mode', label: 'Whisper mode', type: 'select', default: 'transcribe',
        visibleWhen: (mid) => /whisper/i.test(mid || ''),
        help: 'Translate forces output to English regardless of source language. Whisper-only.',
        options: [
          { value: 'transcribe', label: 'Transcribe (preserve source language)' },
          { value: 'translate',  label: 'Translate to English' },
        ],
      },
      { key: 'chunk_length_s',    label: 'Chunk length (s)',  type: 'number',  default: 30,    min: 5,  max: 60, step: 1, help: 'Whisper\'s 30s context window. Only used for long-audio (>30s) variant.' },
      { key: 'stride_length_s',   label: 'Stride length (s)', type: 'number',  default: 5,     min: 0,  max: 15, step: 1, help: 'Overlap between chunks. Helps stitch words across boundaries.' },
      { key: 'return_timestamps', label: 'Return timestamps', type: 'boolean', default: false,                            help: 'Attach per-segment (or per-word) timestamps in the output.' },
    ]},
  'text-to-speech':               { nm: 'Synthesize', input: 'text',  output: 'audio', icon: 'waveform', accent: 'oklch(70% 0.13 65)',
    guide: {
      summary: 'Text-to-speech (SpeechT5, VITS, Bark, MMS-TTS, FastSpeech2, …).',
      rows: [
        { k: 'Text',   v: 'Any length. Long inputs may be chunked',                                              req: true  },
        { k: 'Voice',  v: 'SpeechT5 only: pick a speaker index (CMU-Arctic x-vector). Default 7306.',             req: false },
      ],
    },
    params: [
      { key: 'speaker_index', label: 'Speaker index', type: 'number', default: 7306, min: 0, max: 7930, step: 1,
        visibleWhen: (mid) => /speecht5/i.test(mid || ''),
        help: 'SpeechT5 only. The CMU-Arctic dataset has 7931 x-vectors (0-7930). 7306 is the HF default (clear female voice). Try other indices for different speakers.' },
    ]},

  'text-to-image': { nm: 'Generate image', input: 'text', output: 'image', icon: 'sparkle', accent: 'oklch(70% 0.15 320)',
    guide: {
      summary: 'Text-to-image diffusion (SD, SDXL, FLUX, …).',
      rows: [{ k: 'Prompt', v: 'What to paint. Longer, specific prompts work best', req: true }],
      example: 'a cinematic photo of a red fox curled up on a moss-covered stone, golden-hour lighting, 35mm',
    },
    params: [
      { key: 'num_inference_steps', label: 'Inference steps',  type: 'number', default: 20,  min: 1,  max: 100, step: 1,   help: 'More steps → higher quality, linearly slower. 20-30 is a sweet spot for most SDXL / SD models; FLUX works at 4-8.' },
      { key: 'guidance_scale',      label: 'Guidance scale',   type: 'range',  default: 7.5, min: 0,  max: 20,  step: 0.5, help: 'Classifier-free guidance. Higher = stricter prompt adherence, less diversity.' },
      { key: 'negative_prompt',     label: 'Negative prompt',  type: 'text',   default: '',                                help: 'Things to steer AWAY from. e.g. "blurry, extra fingers, watermark".' },
    ]},
  'image-to-image': { nm: 'Edit image', input: 'image', output: 'image', icon: 'sparkle', accent: 'oklch(70% 0.15 320)',
    textSlot: {
      label: 'Prompt',
      placeholder: 'make it look like an oil painting, warmer lighting, add snow on the roof',
      required: true,
      help: 'How to transform the source image. `strength` controls how much of the original to preserve (see parameters).',
    },
    guide: {
      summary: 'Img2img diffusion. Rewrite a source image guided by a text prompt (SD, SDXL, FLUX, …).',
      rows: [
        { k: 'Image',  v: 'Source JPG/PNG/WebP',                      req: true },
        { k: 'Prompt', v: 'What to change / the target style',         req: true },
      ],
      example: 'cyberpunk cityscape at night, neon reflections, heavy rain',
    },
    params: [
      { key: 'strength',            label: 'Strength',         type: 'range',  default: 0.8, min: 0,  max: 1,   step: 0.05, help: 'How much to transform: 0 = original image unchanged, 1 = fully regenerated from prompt.' },
      { key: 'num_inference_steps', label: 'Inference steps',  type: 'number', default: 20,  min: 1,  max: 100, step: 1,    help: 'More steps → higher quality, linearly slower.' },
      { key: 'guidance_scale',      label: 'Guidance scale',   type: 'range',  default: 7.5, min: 0,  max: 20,  step: 0.5,  help: 'Classifier-free guidance. Higher = stricter prompt adherence.' },
      { key: 'negative_prompt',     label: 'Negative prompt',  type: 'text',   default: '',                                help: 'Things to steer AWAY from.' },
    ]},

  'text-generation':     { nm: 'Generate',  input: 'text', output: 'text',   icon: 'chat', accent: 'oklch(70% 0.12 230)',
    guide: {
      summary: 'Causal LM (Llama, Qwen, Mistral, Phi, Gemma, …).',
      rows: [{ k: 'Prompt', v: 'Any text. The model continues from it', req: true }],
    },
    params: GEN_PARAMS},
  'text2text-generation':{ nm: 'Rewrite',   input: 'text', output: 'text',   icon: 'chat', accent: 'oklch(70% 0.12 230)',
    guide: {
      summary: 'Encoder-decoder seq2seq rewrite.',
      rows: [{ k: 'Text', v: 'Input sequence to transform', req: true }],
    },
    params: GEN_PARAMS},
  'translation':         { nm: 'Translate', input: 'text', output: 'text',   icon: 'chat', accent: 'oklch(70% 0.12 230)',
    guide: {
      summary: 'Translation model (Marian, NLLB, M2M-100, FSMT).',
      rows: [
        { k: 'Text',     v: 'Sentence or paragraph in the source language',                          req: true  },
        { k: 'Src / Tgt', v: 'NLLB and M2M-100 need language codes (see params). Marian is fixed-pair.', req: false },
      ],
    },
    params: [
      ...GEN_PARAMS,
      { key: 'src_lang', label: 'Source language', type: 'text', default: '',
        help: 'NLLB and M2M-100 only. NLLB codes: "eng_Latn", "fra_Latn", "hin_Deva". M2M-100 codes: "en", "fr". Leave empty for Marian (fixed-pair).' },
      { key: 'tgt_lang', label: 'Target language', type: 'text', default: '',
        help: 'Same format as source. Check the model card for valid codes.' },
    ]},
  'summarization':       { nm: 'Summarize', input: 'text', output: 'text',   icon: 'chat', accent: 'oklch(70% 0.12 230)',
    guide: {
      summary: 'Abstractive summarizer.',
      rows: [{ k: 'Text', v: 'The passage to summarize', req: true }],
    },
    params: GEN_PARAMS},
  'feature-extraction':  { nm: 'Embed',     input: 'text', output: 'vector', icon: 'embed', accent: 'oklch(70% 0.10 250)',
    guide: {
      summary: 'Text embeddings (sentence-transformers, BGE, E5, GTE, MiniLM, Nomic, …).',
      rows: [{ k: 'Text', v: 'A sentence or passage to embed', req: true }],
    },
    params: [
      { key: 'normalize',  label: 'Normalize (unit length)', type: 'boolean', default: true,
        help: 'L2-normalize the vector so cosine similarity is a plain dot product. Cosine similarity then reduces to a plain dot product.' },
      { key: 'dimensions', label: 'Dimensions', type: 'number', default: 0, min: 0, max: 4096, step: 1,
        help: 'Truncate to the first N dimensions (Matryoshka models like nomic-embed / text-embedding-3). 0 = full size.' },
    ]},
  'sentence-similarity': { nm: 'Embed',     input: 'text', output: 'vector', icon: 'embed', accent: 'oklch(70% 0.10 250)',
    guide: {
      summary: 'Text embeddings (sentence-transformers, BGE, E5, GTE, MiniLM, Nomic, …).',
      rows: [{ k: 'Text', v: 'A sentence or passage to embed', req: true }],
    },
    params: [
      { key: 'normalize',  label: 'Normalize (unit length)', type: 'boolean', default: true,
        help: 'L2-normalize the vector so cosine similarity is a plain dot product. Cosine similarity then reduces to a plain dot product.' },
      { key: 'dimensions', label: 'Dimensions', type: 'number', default: 0, min: 0, max: 4096, step: 1,
        help: 'Truncate to the first N dimensions (Matryoshka models like nomic-embed / text-embedding-3). 0 = full size.' },
    ]},
};

function resolveTaskMeta(task) {
  return TASK_META[task] || { nm: task || 'Run', input: 'text', output: 'text', icon: 'cube', accent: 'oklch(70% 0.10 250)' };
}

function titleForTask(meta, input) {
  const verb = meta?.nm || 'Run';
  if (!input) return verb;
  if (input.kind === 'text') {
    const text = (input.text || '').trim().replace(/\s+/g, ' ');
    if (!text) return verb;
    const snippet = text.length > 48 ? text.slice(0, 48).trimEnd() + '…' : text;
    return `${verb}: ${snippet}`;
  }

  const promptText = (input.text || '').trim().replace(/\s+/g, ' ');
  if (promptText) {
    const s = promptText.length > 40 ? promptText.slice(0, 40).trimEnd() + '…' : promptText;
    return `${verb}: ${s}`;
  }
  const raw = input.name || '';
  const base = raw.replace(/\.[^./\\]+$/, '').replace(/[_\-]+/g, ' ').trim() || raw || 'input';
  const short = base.length > 40 ? base.slice(0, 40).trimEnd() + '…' : base;
  return `${verb} · ${short}`;
}

const FLORENCE_TASKS = [
  { value: '<CAPTION>',                            label: 'Caption (short)',         output: 'text',  desc: 'A one-line caption of the image.' },
  { value: '<DETAILED_CAPTION>',                   label: 'Caption (detailed)',      output: 'text',  desc: 'A few-sentence description of the image.' },
  { value: '<MORE_DETAILED_CAPTION>',              label: 'Caption (more detailed)', output: 'text',  desc: 'A long, dense paragraph describing the image.' },
  { value: '<OD>',                                 label: 'Object detection',        output: 'boxes', desc: 'Closed-set detection. Boxes for known classes.' },
  { value: '<DENSE_REGION_CAPTION>',               label: 'Dense region captions',   output: 'boxes', desc: 'Boxes around regions, each with a short caption.' },
  { value: '<REGION_PROPOSAL>',                    label: 'Region proposals',        output: 'boxes', desc: 'Salient region boxes without labels.' },
  { value: '<OCR>',                                label: 'OCR (plain text)',        output: 'text',  desc: 'Reads all text in the image as a single block.' },
  { value: '<OCR_WITH_REGION>',                    label: 'OCR with regions',        output: 'boxes', desc: 'Reads text and returns a box per word/line.' },
  { value: '<CAPTION_TO_PHRASE_GROUNDING>',        label: 'Phrase grounding',        output: 'boxes', usesPrompt: true, desc: 'Locate the phrase you type in the image.' },
  { value: '<REFERRING_EXPRESSION_SEGMENTATION>',  label: 'Referring segmentation',  output: 'text',  usesPrompt: true, desc: 'Polygons for the thing you describe.' },
  { value: '<OPEN_VOCABULARY_DETECTION>',          label: 'Open-vocab detection',    output: 'boxes', usesPrompt: true, desc: 'Detect arbitrary classes you list in the prompt.' },
];

function FlorenceTaskBar({ value, onChange }) {
  const [open, setOpen] = useStateTW(false);
  const wrapRef = useRefTW(null);
  const cur = FLORENCE_TASKS.find(t => t.value === value) || FLORENCE_TASKS[0];

  useEffectTW(() => {
    if (!open) return;
    const onDoc = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div className={`florence-pick ${open ? 'open' : ''}`} ref={wrapRef}>
      <button
        type="button"
        className="florence-pick-trigger"
        onClick={() => setOpen(o => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <Icon name="sparkle" size={12}/>
        <span className="florence-pick-k">Florence-2 task</span>
        <span className="florence-pick-sep">·</span>
        <span className="florence-pick-cur">{cur.label}</span>
        {cur.usesPrompt && <span className="florence-pick-tag">uses prompt</span>}
        <span style={{flex: 1}}/>
        <Icon name="chevron" size={11}/>
      </button>
      {open && (
        <div className="florence-pick-menu" role="listbox">
          {FLORENCE_TASKS.map(t => (
            <button
              key={t.value}
              type="button"
              role="option"
              aria-selected={value === t.value}
              className={`florence-pick-opt ${value === t.value ? 'active' : ''}`}
              onClick={() => { onChange(t.value); setOpen(false); }}
            >
              <span className="florence-pick-opt-row">
                <span className="florence-pick-opt-l">{t.label}</span>
                {t.usesPrompt && <span className="florence-pick-opt-tag">+ prompt</span>}
                {value === t.value && <Icon name="check" size={11}/>}
              </span>
              <span className="florence-pick-opt-d">{t.desc}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

const WHISPER_MODES = [
  { value: 'transcribe', label: 'Transcribe',           desc: 'Preserve source language' },
  { value: 'translate',  label: 'Translate to English', desc: 'Force English output' },
];

function WhisperModeBar({ value, onChange }) {
  return (
    <div className="whisper-bar">
      <div className="whisper-bar-head">
        <Icon name="waveform" size={11}/>
        <span className="whisper-bar-k">Whisper mode</span>
      </div>
      <div className="whisper-bar-toggle" role="radiogroup" aria-label="Whisper mode">
        {WHISPER_MODES.map(m => (
          <button
            key={m.value}
            type="button"
            role="radio"
            aria-checked={value === m.value}
            className={`whisper-pill ${value === m.value ? 'active' : ''}`}
            onClick={() => onChange(m.value)}
            title={m.desc}
          >
            <span className="whisper-pill-l">{m.label}</span>
            <span className="whisper-pill-d">{m.desc}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

const ONEFORMER_MODES = [
  { value: 'semantic', label: 'Semantic', desc: 'One mask per class' },
  { value: 'instance', label: 'Instance', desc: 'One mask per object' },
  { value: 'panoptic', label: 'Panoptic', desc: 'Stuff classes + thing instances' },
];

function OneFormerModeBar({ value, onChange }) {
  return (
    <div className="whisper-bar">
      <div className="whisper-bar-head">
        <Icon name="eye" size={11}/>
        <span className="whisper-bar-k">OneFormer mode</span>
      </div>
      <div className="whisper-bar-toggle" role="radiogroup" aria-label="OneFormer mode">
        {ONEFORMER_MODES.map(m => (
          <button
            key={m.value}
            type="button"
            role="radio"
            aria-checked={value === m.value}
            className={`whisper-pill ${value === m.value ? 'active' : ''}`}
            onClick={() => onChange(m.value)}
            title={m.desc}
          >
            <span className="whisper-pill-l">{m.label}</span>
            <span className="whisper-pill-d">{m.desc}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function FlorenceEmpty({ florenceTask, modelId, accent }) {
  const cur = FLORENCE_TASKS.find(t => t.value === florenceTask) || FLORENCE_TASKS[0];
  return (
    <div className="florence-empty">
      <div className="florence-empty-icon" style={{color: accent}}>
        <Icon name="sparkle" size={22} stroke={1.4}/>
      </div>
      <div className="florence-empty-task">{cur.label}</div>
      <div className="florence-empty-desc">{cur.desc}</div>
      <div className="florence-empty-hint">
        {cur.usesPrompt
          ? 'Pick a task above, attach an image, type the phrase, then run.'
          : 'Pick a task above, attach an image, then run.'}
      </div>
      <div className="florence-empty-mid mono">{modelId}</div>
    </div>
  );
}

function TaskWorkspace({ sessionId, modelId, modelMeta, onSaved }) {
  const [session, setSession] = useStateTW(null);
  const [textInput, setTextInput] = useStateTW('');
  const [fileInput, setFileInput] = useStateTW(null); 
  const [running, setRunning] = useStateTW(false);


  const [stopping, setStopping] = useStateTW(false);


  const stoppedByUserRef = useRefTW(false);
  const stop = async () => {
    if (!running || stopping) return;
    setStopping(true);
    stoppedByUserRef.current = true;
    try { await window.inferml?.tasks?.stop?.(); } catch {}
  };
  const [error, setError] = useStateTW(null);
  const [paramValues, setParamValues] = useStateTW({});

  const [samPoints, setSamPoints] = useStateTW([]);
  const [samMode, setSamMode] = useStateTW(1); 
  const scrollRef = useRefTW(null);
  const paramsTaskRef = useRefTW('');

  const resolvedTask = session?.task || modelMeta?.task || '';
  const meta = resolveTaskMeta(resolvedTask);
  const isFlorence = /florence-?2/i.test(modelId || '');
  const florenceTask = paramValues.florence_task || '<CAPTION>';
  const florenceMeta = isFlorence ? FLORENCE_TASKS.find(t => t.value === florenceTask) : null;
  const florenceUsesPrompt = !!florenceMeta?.usesPrompt;
  const isWhisper = /whisper/i.test(modelId || '') && resolvedTask === 'automatic-speech-recognition';
  const whisperMode = paramValues.whisper_mode || 'transcribe';
  const isOneFormer = /oneformer/i.test(modelId || '') && resolvedTask === 'image-segmentation';
  const oneformerMode = paramValues.oneformer_mode || 'semantic';



  useEffectTW(() => {
    if (!resolvedTask || resolvedTask === paramsTaskRef.current) return;
    paramsTaskRef.current = resolvedTask;
    const schema = meta.params || [];
    const visible = schema.filter(p => !p.visibleWhen || p.visibleWhen(modelId || ''));
    const defaults = Object.fromEntries(visible.map(p => [p.key, p.default]));
    setParamValues(defaults);
  }, [resolvedTask, modelId]);

  useEffectTW(() => {
    let cancelled = false;
    (async () => {
      if (!sessionId) return;
      const s = await window.inferml.chats.get(sessionId);
      if (cancelled || !s) return;

      let task = s.task;
      if (!task && s.modelId) {
        const installed = await window.inferml?.hf.installed().catch(() => null);
        task = installed?.[s.modelId]?.task || modelMeta?.task || '';
      }

      const runs = (s.runs || []).map(r => r.status === 'stub' || (r.status === 'running' && !r.output)
        ? { ...r, status: 'pending', output: null }
        : r);
      const next = { ...s, runs, task, running: false };
      const changed = task !== s.task || runs.some((r, i) => r !== (s.runs || [])[i]);
      if (changed) window.inferml.chats.save(next).catch(() => {});
      if (!cancelled) setSession(next);
    })();
    return () => { cancelled = true; };
  }, [sessionId]);

  useEffectTW(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [session?.runs?.length, running]);

  if (!session) return <div className="tw"><div className="tw-body"><div className="chat-empty">Loading…</div></div></div>;

  const pickImage = async () => {
    const att = await window.inferml.dialog.openImage();
    if (att) {
      setFileInput(att);
      setSamPoints([]); 
    }
  };
  const pickAudio = async () => {
    const att = await window.inferml.dialog.openAudio();
    if (att) setFileInput(att);
  };

  const textSlot = meta.textSlot;
  const textSlotFilled = textInput.trim().length > 0;


  const promptRequired = !!textSlot && (textSlot.required || (isFlorence && florenceUsesPrompt));
  const textSlotSatisfied = !textSlot || !promptRequired || textSlotFilled;
  const canRun =
    (meta.input === 'text'  && textInput.trim().length > 0) ||
    (meta.input === 'image' && fileInput?.kind === 'image' && textSlotSatisfied) ||
    (meta.input === 'audio' && fileInput?.kind === 'audio');

  const run = async () => {
    if (!canRun || running) return;
    if (!modelId) {
      setError('This session has no model attached. Delete it and open the model from the sidebar again.');
      return;
    }
    setRunning(true);
    setError(null);




    let input;
    if (meta.input === 'text') {
      input = { kind: 'text', text: textInput.trim() };
    } else if (fileInput) {
      input = { ...fileInput };


      if (textSlot && (!isFlorence || florenceUsesPrompt)) {
        input.text = textInput.trim();
      }
      if (resolvedTask === 'mask-generation' && samPoints.length > 0) {
        input.points = samPoints;
      }
    } else {
      input = fileInput;
    }

    const runId = 'r-' + Math.random().toString(36).slice(2);
    const pending = {
      id: runId,
      input,
      output: null,
      status: 'running',
      ts: Date.now(),
    };
    const nextRuns = [...(session.runs || []), pending];
    const title = session.title && session.title !== 'New session'
      ? session.title
      : titleForTask(meta, input);
    const runningRun = { ...pending, status: 'running' };
    const nextSession = {
      ...session,
      title,
      runs: nextRuns.map(r => r.id === runId ? runningRun : r),
      sub: `${nextRuns.length} run${nextRuns.length === 1 ? '' : 's'}`,
      running: true,
    };
    setSession(nextSession);
    try { await window.inferml.chats.save(nextSession); } catch {}
    onSaved && onSaved(nextSession);
    setTextInput('');
    setFileInput(null);


    const paramsToSend = {};
    for (const p of (meta.params || [])) {
      if (p.visibleWhen && !p.visibleWhen(modelId || '')) continue;
      const v = paramValues[p.key];
      if (v === '' || v === undefined || v === null) continue;
      paramsToSend[p.key] = v;
    }
    const res = await window.inferml.tasks.run({ task: resolvedTask, modelId, input, params: paramsToSend }).catch(e => ({ ok: false, error: String(e?.message || e) }));

    const wasCancelled = stoppedByUserRef.current;
    stoppedByUserRef.current = false;
    const done = {
      ...nextSession,
      running: false,
      runs: nextSession.runs.map(r => {
        if (r.id !== runId) return r;
        if (res?.ok) return { ...r, status: 'done', output: res.output };
        if (wasCancelled) return { ...r, status: 'cancelled', error: 'Stopped by user', output: null };
        return { ...r, status: 'error', error: res?.error || 'inference failed', output: null };
      }),
    };
    setSession(done);
    try { await window.inferml.chats.save(done); } catch {}
    onSaved && onSaved(done);
    setRunning(false);
    setStopping(false);
  };

  const visibleRuns = session.runs || [];

  return (
    <div className="tw">
      <div className="tw-head">
        <div className="tw-tag" style={{color: meta.accent, borderColor: `color-mix(in oklab, ${meta.accent} 40%, transparent)`}}>{meta.nm}</div>
        <div className="tw-head-titles">
          <div className="chat-title">{session.title || 'New session'}</div>
          <div className="chat-sub">{modelId} · {modelMeta?.task || 'task'} · {visibleRuns.length} run{visibleRuns.length === 1 ? '' : 's'}</div>
        </div>
        <div style={{flex:1}}/>
        {modelId && (
          <button
            className="tb-btn"
            onClick={() => window.inferml?.app.openExternal(`https://huggingface.co/${modelId}`)}
            title={`Open ${modelId} on Hugging Face`}
          >
            <Icon name="arrow_right" size={12}/> View on HF
          </button>
        )}
      </div>

      <div ref={scrollRef} className="tw-body">
        {visibleRuns.length === 0 && isFlorence && (
          <FlorenceEmpty florenceTask={florenceTask} modelId={modelId} accent={meta.accent}/>
        )}
        {visibleRuns.length === 0 && !isFlorence && (
          <InputGuide meta={meta} modelId={modelId} onUseExample={(text) => setTextInput(text)}/>
        )}
        {visibleRuns.map(r => <RunCard key={r.id} run={r} meta={meta} modelId={modelId}/>)}
        {error && <div className="chat-err"><Icon name="alert" size={12}/> {error}</div>}
      </div>

      <div className="tw-composer">
        {isFlorence && (
          <FlorenceTaskBar
            value={paramValues.florence_task ?? '<CAPTION>'}
            onChange={(v) => setParamValues(prev => ({ ...prev, florence_task: v }))}
          />
        )}
        {isWhisper && (
          <WhisperModeBar
            value={whisperMode}
            onChange={(v) => setParamValues(prev => ({ ...prev, whisper_mode: v }))}
          />
        )}
        {isOneFormer && (
          <OneFormerModeBar
            value={oneformerMode}
            onChange={(v) => setParamValues(prev => ({ ...prev, oneformer_mode: v }))}
          />
        )}
        {meta.input === 'image' && resolvedTask === 'mask-generation' && fileInput?.kind === 'image' ? (
          <SamPointPicker
            dataUrl={fileInput.dataUrl}
            name={fileInput.name}
            points={samPoints}
            mode={samMode}
            onAddPoint={(pt) => setSamPoints(prev => [...prev, pt])}
            onClearPoints={() => setSamPoints([])}
            onChangeMode={setSamMode}
            onReplace={() => { setFileInput(null); setSamPoints([]); }}
          />
        ) : meta.input === 'image' && textSlot ? (


          (() => {
            const showPrompt = !isFlorence || florenceUsesPrompt;
            const promptRequired = isFlorence ? florenceUsesPrompt : textSlot.required;
            const placeholder = isFlorence
              ? (florenceUsesPrompt ? 'Phrase to ground / detect / segment…' : '')
              : textSlot.placeholder;
            return (
              <div className={`tw-fused ${showPrompt ? '' : 'no-prompt'}`}>
                <div className="tw-fused-top">
                  {fileInput?.kind === 'image' ? (
                    <div className="tw-fused-chip">
                      <img src={fileInput.dataUrl} alt={fileInput.name}/>
                      <span className="tw-fused-chip-nm">{fileInput.name}</span>
                      <button
                        type="button"
                        className="tw-fused-chip-x"
                        onClick={() => setFileInput(null)}
                        aria-label="Remove image"
                      >
                        <Icon name="x" size={11}/>
                      </button>
                    </div>
                  ) : (
                    <button type="button" className="tw-fused-attach" onClick={pickImage}>
                      <Icon name="paperclip" size={13}/>
                      <span>Attach image</span>
                      <span className="tw-fused-attach-req">required</span>
                    </button>
                  )}
                  {!showPrompt && (
                    <span className="tw-fused-noprompt">This task doesn't use a prompt. Run with just the image.</span>
                  )}
                  <div style={{flex: 1}}/>
                  {showPrompt && (promptRequired
                    ? <span className="tw-text-slot-req">prompt required</span>
                    : <span className="tw-text-slot-opt">prompt optional</span>)}
                  {!showPrompt && (
                    <button
                      type="button"
                      className={`cc-send tw-fused-top-run ${running ? 'is-stop' : ''}`}
                      onClick={running ? stop : run}
                      disabled={running ? stopping : !canRun}
                    >
                      {running ? (stopping ? 'Stopping…' : <><Icon name="x" size={11}/> Stop</>) : 'Run'}
                    </button>
                  )}
                </div>
                {showPrompt && (
                  <div className="tw-fused-row">
                    <textarea
                      className="tw-fused-input"
                      rows={1}
                      placeholder={placeholder}
                      value={textInput}
                      onChange={e => setTextInput(e.target.value)}
                      onKeyDown={e => {
                        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run(); }
                      }}
                    />
                    <button
                      type="button"
                      className={`cc-send ${running ? 'is-stop' : ''}`}
                      onClick={running ? stop : run}
                      disabled={running ? stopping : !canRun}
                    >
                      {running ? (stopping ? 'Stopping…' : <><Icon name="x" size={11}/> Stop</>) : 'Run'}
                    </button>
                  </div>
                )}
                {showPrompt && textSlot.help && !isFlorence && <div className="tw-fused-help">{textSlot.help}</div>}
              </div>
            );
          })()
        ) : meta.input === 'image' && (
          <div className="tw-input-row">
            <div className="tw-input-panel" onClick={pickImage}>
              {fileInput?.kind === 'image'
                ? <div className="tw-preview"><img src={fileInput.dataUrl} alt={fileInput.name}/><span className="tw-preview-nm">{fileInput.name}</span><button className="cc-att-x" onClick={(e) => { e.stopPropagation(); setFileInput(null); }}><Icon name="x" size={11}/></button></div>
                : <div className="tw-drop"><Icon name="paperclip" size={16}/><span>Click to choose an image</span></div>}
            </div>
            <button
              type="button"
              className={`cc-send tw-row-run ${running ? 'is-stop' : ''}`}
              onClick={running ? stop : run}
              disabled={running ? stopping : !canRun}
            >
              {running ? (stopping ? 'Stopping…' : <><Icon name="x" size={11}/> Stop</>) : 'Run'}
            </button>
          </div>
        )}
        {meta.input === 'audio' && (
          <div className="tw-input-row">
            <div className="tw-input-panel" onClick={pickAudio}>
              {fileInput?.kind === 'audio'
                ? <div className="tw-preview audio"><Icon name="waveform" size={16}/><span className="tw-preview-nm">{fileInput.name}</span><button className="cc-att-x" onClick={(e) => { e.stopPropagation(); setFileInput(null); }}><Icon name="x" size={11}/></button></div>
                : <div className="tw-drop"><Icon name="mic" size={16}/><span>Click to choose an audio file</span></div>}
            </div>
            <button
              type="button"
              className={`cc-send tw-row-run ${running ? 'is-stop' : ''}`}
              onClick={running ? stop : run}
              disabled={running ? stopping : !canRun}
            >
              {running ? (stopping ? 'Stopping…' : <><Icon name="x" size={11}/> Stop</>) : 'Run'}
            </button>
          </div>
        )}
        {meta.input === 'text' && (
          <div className="tw-fused-row">
            <textarea
              className="cc-input cc-input-inline"
              placeholder={
                meta.output === 'image' ? 'Describe what to generate…'
                : meta.output === 'audio' ? 'Enter text to synthesize…'
                : 'Enter text…'
              }
              rows={1}
              value={textInput}
              onChange={e => setTextInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run(); }
              }}
            />
            <button
              type="button"
              className={`cc-send ${running ? 'is-stop' : ''}`}
              onClick={running ? stop : run}
              disabled={running ? stopping : !canRun}
            >
              {running ? (stopping ? 'Stopping…' : <><Icon name="x" size={11}/> Stop</>) : 'Run'}
            </button>
          </div>
        )}
        <ParamsPanel
          schema={meta.params}
          values={paramValues}
          setValues={setParamValues}
          modelId={modelId}
          hideKeys={[
            ...(isFlorence ? ['florence_task'] : []),
            ...(isWhisper ? ['whisper_mode'] : []),
            ...(isOneFormer ? ['oneformer_mode'] : []),
          ]}
        />
        {/* Every composer mode now renders its own inline Run except the SAM
            point-picker (whose interactive image needs the full row width).
            Show cc-foot only there. */}
        {meta.input === 'image' && resolvedTask === 'mask-generation' && fileInput?.kind === 'image' && (
          <div className="cc-foot">
            <div style={{flex:1}}/>
            <button
              className={`cc-send ${running ? 'is-stop' : ''}`}
              onClick={running ? stop : run}
              disabled={running ? stopping : !canRun}
            >
              {running
                ? (stopping ? 'Stopping…' : <><Icon name="x" size={11}/> Stop</>)
                : <>Run <span className="cc-kbd">⌘↵</span></>}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function SamPointPicker({ dataUrl, name, points, mode, onAddPoint, onClearPoints, onChangeMode, onReplace }) {
  const imgRef = useRefTW(null);
  const handleClick = (e) => {
    const img = imgRef.current;
    if (!img) return;
    const rect = img.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    onAddPoint({ x, y, label: mode });
  };
  return (
    <div className="sam-picker">
      <div className="sam-canvas">
        <div className="sam-image-wrap" onClick={handleClick}>
          <img ref={imgRef} src={dataUrl} alt={name || 'input'} draggable={false}/>
          {points.map((p, i) => (
            <span
              key={i}
              className={`sam-point ${p.label === 1 ? 'fg' : 'bg'}`}
              style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }}
              title={`${p.label === 1 ? 'include' : 'exclude'} · ${i + 1}`}
            >{i + 1}</span>
          ))}
        </div>
        <div className="sam-canvas-hint">
          {points.length === 0
            ? 'Click to add points, or run as-is for auto-grid segmentation.'
            : `${points.length} point${points.length === 1 ? '' : 's'} set. Run to segment.`}
        </div>
      </div>
      <div className="sam-picker-controls">
        <div className="sam-mode-toggle">
          <button
            type="button"
            className={mode === 1 ? 'active' : ''}
            onClick={() => onChangeMode(1)}
            title="Click adds an INCLUDE point"
          >
            <span className="sam-mode-dot fg"/> Include
          </button>
          <button
            type="button"
            className={mode === 0 ? 'active' : ''}
            onClick={() => onChangeMode(0)}
            title="Click adds an EXCLUDE point"
          >
            <span className="sam-mode-dot bg"/> Exclude
          </button>
        </div>
        <span className="sam-point-count mono">
          {points.length} point{points.length === 1 ? '' : 's'}
        </span>
        <div style={{flex: 1}}/>
        {points.length > 0 && (
          <button type="button" className="tb-btn" onClick={onClearPoints}>
            <Icon name="x" size={11}/> Clear
          </button>
        )}
        <button type="button" className="tb-btn" onClick={onReplace}>
          <Icon name="paperclip" size={11}/> Replace image
        </button>
      </div>
    </div>
  );
}

function fmtParamValue(p, v) {
  if (p.type === 'boolean') return v ? 'on' : 'off';
  if (p.type === 'select') {
    const opt = (p.options || []).find(o => o.value === v);
    return opt ? opt.label : String(v ?? '');
  }
  if (p.type === 'range' || p.type === 'number') {
    if (v === '' || v === undefined || v === null) return '-';
    const n = Number(v);
    if (!Number.isFinite(n)) return '-';
    // Show decimals only if the step is fractional.
    const dec = p.step != null && p.step < 1 ? (String(p.step).split('.')[1]?.length || 2) : 0;
    return n.toFixed(dec);
  }
  return String(v ?? '');
}

function ParamsPanel({ schema, values, setValues, modelId, hideKeys }) {
  if (!schema || schema.length === 0) return null;
  let visible = schema.filter(p => !p.visibleWhen || p.visibleWhen(modelId || ''));
  if (hideKeys && hideKeys.length) {
    const hide = new Set(hideKeys);
    visible = visible.filter(p => !hide.has(p.key));
  }
  if (visible.length === 0) return null;

  const changedCount = visible.reduce((n, p) => {
    const v = values[p.key];
    return n + (v !== undefined && v !== p.default ? 1 : 0);
  }, 0);

  const setKey = (k, v) => setValues(prev => ({ ...prev, [k]: v }));
  const reset = () => {
    const defaults = Object.fromEntries(visible.map(p => [p.key, p.default]));
    setValues(prev => ({ ...prev, ...defaults }));
  };

  return (
    <details className="tw-params">
      <summary className="tw-params-head">
        <Icon name="settings" size={12}/>
        <span className="tw-params-title">Parameters</span>
        {changedCount > 0 && <span className="tw-params-changed">{changedCount} changed</span>}
        <span style={{flex: 1}}/>
        <span className="tw-params-hint">{visible.length} available</span>
        <span className="tw-params-caret"><Icon name="chevron" size={12}/></span>
      </summary>
      <div className="tw-params-body">
        {visible.map(p => (
          <ParamRow key={p.key} param={p} value={values[p.key]} onChange={(v) => setKey(p.key, v)}/>
        ))}
        {changedCount > 0 && (
          <button type="button" className="tw-params-reset" onClick={reset}>
            <Icon name="x" size={10}/> Reset to defaults
          </button>
        )}
      </div>
    </details>
  );
}

function ParamRow({ param, value, onChange }) {
  const id = `p-${param.key}`;
  return (
    <div className="tw-param">
      <div className="tw-param-label">
        <label htmlFor={id}>{param.label}</label>
        {(param.type === 'range' || param.type === 'number') && (
          <span className="tw-param-value mono">{fmtParamValue(param, value)}</span>
        )}
      </div>
      <div className="tw-param-control">
        {param.type === 'number' && (
          <input
            id={id}
            type="number"
            min={param.min}
            max={param.max}
            step={param.step}
            value={value ?? ''}
            onChange={e => onChange(e.target.value === '' ? '' : Number(e.target.value))}
          />
        )}
        {param.type === 'range' && (
          <input
            id={id}
            type="range"
            min={param.min}
            max={param.max}
            step={param.step}
            value={value ?? param.default}
            onChange={e => onChange(Number(e.target.value))}
          />
        )}
        {param.type === 'text' && (
          <input
            id={id}
            type="text"
            placeholder={param.placeholder || ''}
            value={value ?? ''}
            onChange={e => onChange(e.target.value)}
          />
        )}
        {param.type === 'boolean' && (
          <label className="tw-param-switch" htmlFor={id}>
            <input
              id={id}
              type="checkbox"
              checked={!!value}
              onChange={e => onChange(e.target.checked)}
            />
            <span className="tw-param-switch-track"/>
            <span className="tw-param-switch-label">{value ? 'on' : 'off'}</span>
          </label>
        )}
        {param.type === 'select' && (
          <select
            id={id}
            value={value ?? param.default}
            onChange={e => onChange(e.target.value)}
          >
            {param.options.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        )}
      </div>
      {param.help && <div className="tw-param-help">{param.help}</div>}
    </div>
  );
}

function InputGuide({ meta, modelId, onUseExample }) {
  const guide = meta.guide;
  return (
    <div className="chat-empty guide">
      <div className="chat-empty-ic" style={{color: meta.accent}}>
        <Icon name={meta.icon} size={28} stroke={1.4}/>
      </div>
      <div className="chat-empty-t">{meta.nm}</div>
      {guide?.summary && <div className="chat-empty-s">{guide.summary}</div>}

      {guide?.rows?.length > 0 && (
        <div className="tw-guide">
          <div className="tw-guide-h">What this model expects</div>
          <div className="tw-guide-rows">
            {guide.rows.map((r, i) => (
              <div key={i} className="tw-guide-row">
                <span className="tw-guide-k">{r.k}</span>
                <span className={`tw-guide-req ${r.req ? 'req' : 'opt'}`}>
                  {r.req ? 'required' : 'optional'}
                </span>
                <span className="tw-guide-v">{r.v}</span>
              </div>
            ))}
          </div>
          {guide.example && (
            <button
              type="button"
              className="tw-guide-example"
              onClick={() => onUseExample && onUseExample(guide.example)}
              title="Fill the text field with this example"
            >
              <span className="tw-guide-example-k">Example</span>
              <code>{guide.example}</code>
              <span className="tw-guide-example-use">use →</span>
            </button>
          )}
        </div>
      )}

      <div className="tw-guide-model mono">{modelId}</div>
    </div>
  );
}

function RunCard({ run, meta, modelId }) {
  const statusLabel =
    run.status === 'running'   ? 'running…' :
    run.status === 'error'     ? 'failed' :
    run.status === 'cancelled' ? 'stopped' :
    run.status === 'done'      ? 'done' :
    run.status === 'pending'   ? 'pending' :
    run.status;

  return (
    <div className="tw-run">
      <div className="tw-run-head">
        <span className="tw-run-label">Input</span>
        <span className="tw-run-status">{statusLabel}</span>
      </div>
      <div className="tw-run-input">
        {run.input?.kind === 'image' && (
          <div className="tw-run-image-wrap">
            <img src={run.input.dataUrl} alt={run.input.name}/>
            {Array.isArray(run.input.points) && run.input.points.map((p, i) => (
              <span
                key={i}
                className={`sam-point sam-point-sm ${p.label === 1 ? 'fg' : 'bg'}`}
                style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }}
              >{i + 1}</span>
            ))}
          </div>
        )}
        {run.input?.kind === 'audio' && <div className="tw-audio-chip"><Icon name="waveform" size={12}/> {run.input.name}</div>}
        {run.input?.kind === 'text' && <div className="tw-text">{run.input.text}</div>}
        {/* Secondary text slot on image-mode runs - candidate labels or a VLM prompt. */}
        {run.input?.kind === 'image' && run.input?.text && (
          <div className="tw-run-prompt">
            <span className="tw-run-prompt-k">{meta.textSlot?.label || 'Prompt'}</span>
            <span className="tw-run-prompt-v">{run.input.text}</span>
          </div>
        )}
      </div>
      <div className="tw-run-head" style={{marginTop: 10}}>
        <span className="tw-run-label" style={{color: meta.accent}}>Output</span>
        <span className="tw-run-status mono">{modelId}</span>
      </div>
      <div className="tw-run-output">
        {run.status === 'running' && (
          <div className="tw-running">
            <div className="tw-skeleton"/>
            <div className="tw-running-note">First run downloads the model. This can take a minute.</div>
          </div>
        )}
        {run.status === 'error' && (
          <div className="tw-error">
            <Icon name="alert" size={14}/>
            <div className="tw-error-body">
              <div className="tw-error-t">Inference failed</div>
              <div className="tw-error-s">{run.error || 'Something went wrong. Check the log file for details.'}</div>
              <button
                type="button"
                className="tb-btn tw-error-btn"
                onClick={() => window.inferml?.logs?.view?.()}
                title="Open the log file in your editor"
              >
                <Icon name="file" size={11}/> View logs
              </button>
            </div>
          </div>
        )}
        {run.status === 'cancelled' && (
          <div className="tw-cancelled">
            <Icon name="x" size={14}/>
            <span>Stopped by user before completion.</span>
          </div>
        )}
        {run.output && <OutputView output={run.output} meta={meta} input={run.input} modelId={modelId}/>}
      </div>
    </div>
  );
}

function OutputView({ output, meta, input, modelId }) {
  if (output.kind === 'boxes') {
    // Prefer the server-rendered annotated PNG; fall back to SVG for legacy outputs.
    if (output.annotated) {
      return (
        <ServerAnnotatedOutput
          annotated={output.annotated}
          items={output.boxes}
          meta={meta}
          modelId={modelId}
          mode="boxes"
        />
      );
    }
    return (
      <AnnotatedOutput
        input={input}
        items={output.boxes.map(b => ({ label: b.label, score: b.score, box: b.box }))}
        meta={meta}
        modelId={modelId}
        mode="boxes"
      />
    );
  }
  if (output.kind === 'masks') {
    if (output.annotated) {
      return (
        <ServerAnnotatedOutput
          annotated={output.annotated}
          items={output.legend || []}
          meta={meta}
          modelId={modelId}
          mode="masks"
        />
      );
    }
    return (
      <SegmentationOutput
        input={input}
        overlay={output.overlay}
        legend={output.legend || []}
        meta={meta}
        modelId={modelId}
      />
    );
  }
  if (output.kind === 'labels') {
    return (
      <div className="tw-labels">
        {output.labels.map((l, i) => (
          <div key={i} className="tw-label-row">
            <span className="tw-chip" style={{background:`color-mix(in oklab, ${meta.accent} 22%, transparent)`,color: meta.accent}}>{l.label}</span>
            <div className="tw-bar"><div className="tw-bar-fill" style={{width: `${l.score * 100}%`, background: meta.accent}}/></div>
            <span className="tw-conf">{(l.score * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    );
  }
  if (output.kind === 'text') {
    return <div className="tw-output-text">{output.text}</div>;
  }
  if (output.kind === 'image') {
    const saveGenerated = () => {
      if (!output.dataUrl) return;
      const a = document.createElement('a');
      a.href = output.dataUrl;
      a.download = `${modelId.split('/').pop()}-${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    };
    return (
      <div className="tw-output-image">
        {output.dataUrl && <img src={output.dataUrl} alt="generated"/>}
        <div className="tw-anno-actions">
          <div style={{flex:1}}/>
          <button className="mc-btn primary" onClick={saveGenerated} disabled={!output.dataUrl}>
            <Icon name="download" size={12}/> Download PNG
          </button>
        </div>
      </div>
    );
  }
  if (output.kind === 'audio') {
    const saveAudio = () => {
      if (!output.dataUrl) return;
      const a = document.createElement('a');
      a.href = output.dataUrl;
      a.download = `${modelId.split('/').pop()}-${Date.now()}.wav`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    };
    return (
      <div className="tw-output-audio">
        {output.dataUrl && <audio controls src={output.dataUrl} style={{width:'100%'}}/>}
        <div className="tw-anno-actions">
          <div style={{flex:1}}/>
          <button className="mc-btn primary" onClick={saveAudio} disabled={!output.dataUrl}>
            <Icon name="download" size={12}/> Download WAV
          </button>
        </div>
      </div>
    );
  }
  if (output.kind === 'vector') {
    return (
      <div className="tw-vector">
        <div className="tw-anno-count">{output.dim}-dimensional embedding</div>
        <div className="mono" style={{fontSize:10.5,color:'var(--fg-2)',whiteSpace:'pre-wrap',wordBreak:'break-all',marginTop:8}}>
          [{output.sample.map(n => n.toFixed(4)).join(', ')}, …]
        </div>
      </div>
    );
  }
  return null;
}

function SegmentationOutput({ input, overlay, legend, meta, modelId }) {
  const imgRef = useRefTW(null);
  const [dims, setDims] = useStateTW(null);
  const [downloading, setDownloading] = useStateTW(false);

  const onImgLoad = (e) => {
    const n = e.currentTarget;
    setDims({ w: n.naturalWidth, h: n.naturalHeight });
  };

  const handleDownload = async () => {
    if (!input?.dataUrl || !overlay) return;
    setDownloading(true);
    try {
      const [baseImg, overlayImg] = await Promise.all([
        loadImage(input.dataUrl),
        loadImage(overlay),
      ]);
      const canvas = document.createElement('canvas');
      canvas.width = baseImg.naturalWidth;
      canvas.height = baseImg.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(baseImg, 0, 0);
      ctx.drawImage(overlayImg, 0, 0, canvas.width, canvas.height);
      const a = document.createElement('a');
      a.href = canvas.toDataURL('image/png');
      a.download = `${modelId.split('/').pop()}-segmentation-${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="tw-anno-wrap">
      <div className="tw-anno">
        <img ref={imgRef} src={input?.dataUrl} alt={input?.name} onLoad={onImgLoad}/>
        {overlay && <img className="tw-anno-overlay" src={overlay} alt="segmentation overlay"/>}
      </div>
      <div className="tw-anno-actions">
        <span className="tw-anno-count">{legend.length} {legend.length === 1 ? 'class' : 'classes'}</span>
        <div style={{flex:1}}/>
        <button className="mc-btn primary" disabled={downloading} onClick={handleDownload}>
          <Icon name="download" size={12}/> {downloading ? 'Preparing…' : 'Download PNG'}
        </button>
      </div>
      <ul className="tw-legend">
        {legend.map((l, i) => (
          <li key={i}>
            <span className="tw-legend-swatch" style={{background: l.color}}/>
            <span className="tw-legend-label">{l.label}</span>
            {typeof l.coverage === 'number' && <span className="tw-conf">{l.coverage}%</span>}
            {typeof l.score === 'number' && typeof l.coverage !== 'number' && (
              <span className="tw-conf">{(l.score * 100).toFixed(1)}%</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

// Lay out annotation boxes + labels with collision avoidance. Used by both
// the SVG overlay and the canvas download so screen + exported image match.
function layoutAnnotations(items, W, H) {
  const fontSize = Math.max(12, Math.min(W, H) * 0.022);
  const strokeW = Math.max(2, Math.min(W, H) * 0.004);
  const labelH = fontSize + 4;
  // Approximate char width for monospace. 0.6 is conservative for Courier-ish.
  const charW = fontSize * 0.6;

  const raw = items.map((it, idx) => {
    const [nx, ny, nw, nh] = it.box || [0, 0, 0, 0];
    const x = nx * W, y = ny * H;
    const w = nw * W, h = nh * H;
    if (w <= 0 || h <= 0) return null;

    const text = `${it.label || 'object'} ${(it.score * 100).toFixed(0)}%`;
    const labelW = Math.min(text.length * charW + 10, W);

    // Preferred label position: directly above the box, left-aligned with the box.
    // Clamp so it fits within the image horizontally.
    let labelX = Math.max(0, Math.min(x, W - labelW));
    let labelY = y - labelH - 1;
    if (labelY < 0) labelY = y;  // no room above → drop label inside the top of the box

    return { idx, boxX: x, boxY: y, boxW: w, boxH: h, labelX, labelY, labelW, labelH, fontSize, strokeW, text };
  }).filter(Boolean);

  // Sort by desired labelY (top-down) so higher labels get placed first and
  // lower overlapping ones can cascade downward.
  const order = [...raw].sort((a, b) => a.labelY - b.labelY || a.labelX - b.labelX);
  const placed = [];

  const overlaps = (a, b) =>
    a.labelX < b.labelX + b.labelW &&
    a.labelX + a.labelW > b.labelX &&
    a.labelY < b.labelY + b.labelH &&
    a.labelY + a.labelH > b.labelY;

  for (const c of order) {
    // Walk downward in labelH-sized steps until the candidate doesn't collide

    const maxShift = Math.max(0, H - c.labelY - c.labelH);
    const steps = Math.ceil(maxShift / (c.labelH + 1)) + 1;
    for (let i = 0; i < steps; i++) {
      const collision = placed.find(p => overlaps(c, p));
      if (!collision) break;
      c.labelY = collision.labelY + collision.labelH + 1;
      if (c.labelY + c.labelH > H) {
        c.labelY = H - c.labelH;  
        break;
      }
    }
    placed.push(c);
  }

  return raw.sort((a, b) => a.idx - b.idx);
}

function ServerAnnotatedOutput({ annotated, items, meta, modelId, mode }) {
  const download = () => {
    const a = document.createElement('a');
    a.href = annotated;
    a.download = `${modelId.split('/').pop()}-${mode}-${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };
  const count = items.length;
  const noun = mode === 'boxes'
    ? (count === 1 ? 'detection' : 'detections')
    : (count === 1 ? 'class' : 'classes');
  return (
    <div className="tw-anno-wrap">
      <div className="tw-anno">
        <img src={annotated} alt={`${mode} output`}/>
      </div>
      <div className="tw-anno-actions">
        <span className="tw-anno-count">{count} {noun}</span>
        <div style={{flex:1}}/>
        <button className="mc-btn primary" onClick={download}>
          <Icon name="download" size={12}/> Download PNG
        </button>
      </div>
      {mode === 'boxes' && (
        <ul className="tw-list">
          {items.map((it, i) => (
            <li key={i}>
              <span className="tw-chip" style={{background:`color-mix(in oklab, ${meta.accent} 22%, transparent)`, color: meta.accent}}>{it.label}</span>
              <span className="tw-conf">{(it.score * 100).toFixed(1)}%</span>
            </li>
          ))}
        </ul>
      )}
      {mode === 'masks' && (
        <ul className="tw-legend">
          {items.map((l, i) => (
            <li key={i}>
              <span className="tw-legend-swatch" style={{background: l.color}}/>
              <span className="tw-legend-label">{l.label}</span>
              {typeof l.coverage === 'number' && <span className="tw-conf">{l.coverage}%</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function AnnotatedOutput({ input, items, meta, modelId, mode }) {
  const imgRef = useRefTW(null);
  const [dims, setDims] = useStateTW(null);
  const [downloading, setDownloading] = useStateTW(false);

  const onImgLoad = (e) => {
    const n = e.currentTarget;
    setDims({ w: n.naturalWidth, h: n.naturalHeight });
  };

  const layout = React.useMemo(
    () => dims ? layoutAnnotations(items, dims.w, dims.h) : [],
    [items, dims]
  );

  const handleDownload = async () => {
    if (!input?.dataUrl) return;
    setDownloading(true);
    try {
      await downloadAnnotated({
        dataUrl: input.dataUrl,
        items,
        mode,
        accent: meta.accent,
        filename: `${modelId.split('/').pop()}-${mode}-${Date.now()}.png`,
      });
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="tw-anno-wrap">
      <div className="tw-anno">
        <img ref={imgRef} src={input?.dataUrl} alt={input?.name} onLoad={onImgLoad}/>
        {dims && (
          <svg className="tw-anno-svg" viewBox={`0 0 ${dims.w} ${dims.h}`} preserveAspectRatio="none">
            {layout.map((L, i) => (
              <g key={i}>
                {mode === 'masks' && (
                  <rect x={L.boxX} y={L.boxY} width={L.boxW} height={L.boxH} fill={meta.accent} fillOpacity={0.28} stroke="none"/>
                )}
                <rect x={L.boxX} y={L.boxY} width={L.boxW} height={L.boxH} fill="none" stroke={meta.accent} strokeWidth={L.strokeW}/>
                <rect x={L.labelX} y={L.labelY} width={L.labelW} height={L.labelH} fill={meta.accent}/>
                <text
                  x={L.labelX + 5}
                  y={L.labelY + L.fontSize}
                  fontFamily="var(--mono)"
                  fontSize={L.fontSize}
                  fill="#05131a"
                  fontWeight="600"
                >
                  {L.text}
                </text>
              </g>
            ))}
          </svg>
        )}
      </div>
      <div className="tw-anno-actions">
        <span className="tw-anno-count">{items.length} {mode === 'boxes' ? (items.length === 1 ? 'detection' : 'detections') : (items.length === 1 ? 'mask' : 'masks')}</span>
        <div style={{flex:1}}/>
        <button className="mc-btn primary" disabled={downloading} onClick={handleDownload}>
          <Icon name="download" size={12}/> {downloading ? 'Preparing…' : 'Download PNG'}
        </button>
      </div>
      <ul className="tw-list">
        {items.map((it, i) => (
          <li key={i}>
            <span className="tw-chip" style={{background:`color-mix(in oklab, ${meta.accent} 22%, transparent)`, color: meta.accent}}>{it.label}</span>
            <span className="tw-conf">{(it.score * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

async function downloadAnnotated({ dataUrl, items, mode, accent, filename }) {
  const img = new Image();
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = dataUrl; });
  const W = img.naturalWidth, H = img.naturalHeight;
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  const layout = layoutAnnotations(items, W, H);
  const strokeW = Math.max(2, Math.min(W, H) * 0.004);
  ctx.lineWidth = strokeW;
  ctx.strokeStyle = accent;

  for (const L of layout) {
    if (mode === 'masks') {
      ctx.save();
      ctx.globalAlpha = 0.28;
      ctx.fillStyle = accent;
      ctx.fillRect(L.boxX, L.boxY, L.boxW, L.boxH);
      ctx.restore();
    }
    ctx.strokeRect(L.boxX, L.boxY, L.boxW, L.boxH);

    ctx.font = `600 ${L.fontSize}px ui-monospace, Menlo, monospace`;
    ctx.fillStyle = accent;
    ctx.fillRect(L.labelX, L.labelY, L.labelW, L.labelH);
    ctx.fillStyle = '#05131a';
    ctx.fillText(L.text, L.labelX + 5, L.labelY + L.fontSize);
  }

  const url = canvas.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

window.TaskWorkspace = TaskWorkspace;
window.resolveTaskMeta = resolveTaskMeta;
