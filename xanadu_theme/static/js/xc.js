$(document).ready(function(){
  let url_split = window.location.pathname.split("/");
  const page_name = url_split.pop();
  const page_directory = url_split.pop();

  if (page_directory === "demos") {
      const demo_name = page_name.split(".").reverse().pop();
      const xc_url = `https://pennylane.xanadu.ai/lab/tree/demo%3A${demo_name}.ipynb`;

      const button = `
      <a href="${xc_url}" class="button">Open demo in Xanadu Cloud</a>
      `;

      $( button ).insertAfter($("h1").first());
  }
});