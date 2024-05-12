document.getElementById('submit').addEventListener('click', function() {
  // Retrieve form data
  var rating = document.getElementById('rating').value;
  var date = document.getElementById('date').value;
  var variation = document.getElementById('variation').value;
  var review = document.getElementById('review').value;
  
  // Perform further processing (e.g., sending data to server)
  console.log("Rating:", rating);
  console.log("Date:", date);
  console.log("Variation:", variation);
  console.log("Review:", review);

  // You can implement AJAX request or other logic here to submit the data
  // For demonstration, we log the data to console
  alert("Review submitted successfully!");
});