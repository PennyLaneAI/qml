$(document).ready(function(){
  const demo_name = window.location.pathname.split("/").pop().split(".").reverse().pop();
  const xc_url = `https://pennylane.xanadu.ai/lab/tree/demo%3A${demo_name}.ipynb`;

  const button = `
  <button onclick="window.location.href='${xc_url}';">
    Open demo in Xanadu Cloud
  </button>
  `;

  $( button ).insertAfter($("h1").first());
});