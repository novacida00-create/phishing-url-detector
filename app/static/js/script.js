const form = document.getElementById("url-form");
const input = document.getElementById("url");

if (form && input) {
  form.addEventListener("submit", (event) => {
    const value = input.value.trim();

    if (!value) {
      event.preventDefault();
      input.focus();
      return;
    }

    input.value = value;

    const button = document.getElementById("submit-button");
    if (button) {
      button.disabled = true;
      button.textContent = "Memeriksa…";
    }
  });
}

const copyButton = document.getElementById("copy-url");
if (copyButton) {
  copyButton.addEventListener("click", async () => {
    const text = document.getElementById("checked-url")?.textContent.trim() || "";
    try {
      await navigator.clipboard.writeText(text);
      copyButton.textContent = "Tersalin";
    } catch (error) {
      copyButton.textContent = "Gagal";
    }
    setTimeout(() => {
      copyButton.textContent = "Salin";
    }, 1800);
  });
}
