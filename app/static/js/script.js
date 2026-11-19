// Validasi sisi klien sesederhana mungkin; validasi tetap dilakukan di server.
document.getElementById("url-form")?.addEventListener("submit", (event) => {
  const input = document.getElementById("url");
  const value = input.value.trim();

  if (!value) {
    event.preventDefault();
    input.focus();
    return;
  }

  input.value = value;
});
