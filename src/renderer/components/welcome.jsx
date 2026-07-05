function Welcome({ onStart, version }) {
  return (
    <div className="welcome">
      <div className="welcome-bg"/>
      <div className="welcome-inner">
        <Logo size={80}/>
        <div className="welcome-eyebrow">InferML · v{version || '0.1.0'}</div>
        <h1 className="welcome-h1">
          Open-source ML.<br/>
          <span className="accent">Ready to run.</span>
        </h1>
        <p className="welcome-sub">
          Pick any model from the HuggingFace Hub. Install with one click. Runs on CPU or GPU.
        </p>
        <div className="welcome-modal-pills">
          <span>Detection</span>
          <span>Segmentation</span>
          <span>VLMs</span>
          <span>Speech</span>
          <span>Diffusion</span>
          <span>Classification</span>
          <span>Depth</span>
          <span>OCR</span>
        </div>
        <button className="welcome-cta" onClick={onStart}>
          Let's get started
          <Icon name="arrow_right" size={15} stroke={2}/>
        </button>
        <div className="welcome-features">
          <span><span className="dot"/>local inference</span>
          <span><span className="dot"/>any modality</span>
          <span><span className="dot"/>HuggingFace hub</span>
        </div>
      </div>
    </div>
  );
}
window.Welcome = Welcome;
