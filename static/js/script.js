function readURL(input) {
  if (input.files[0]) {
    var reader = new FileReader();
    reader.onload = function (e) {
      $("#upload-form-image").attr("src", e.target.result);
      $("#upload-form-caption").empty();
    };
    reader.readAsDataURL(input.files[0]);
  }
};

$("#upload-form").submit(function(e) {
  e.preventDefault();
  let image = $('#upload-image').prop('files')[0];
  let formData = new FormData();
  if (image) formData.append('image', image);
  formData.append('model', $("#upload-form-model").val())
  // Send image to server
  $.ajax({
    url: 'evaluate/',
    type: 'POST',
    data: formData,
    processData: false,
    contentType: false,
    enctype: 'multipart/form-data',
    beforeSend: function () {
      $("#loader").css("display", "block");
      $("#upload-form-caption").empty();
    },
    success: function(data) {
      $("#upload-form-caption").html(data.text);
      $("#loader").css("display", "none");
    },
    error: function(e) {
      console.log(e)
    }
  })
});

$("#demo-form").submit(function(e) {
  e.preventDefault();

  let formData = new FormData();
  formData.append('model', $("#demo-form-model").val())
  // Send image to server
  $.ajax({
    url: 'evaluate/',
    type: 'POST',
    data: formData,
    processData: false,
    contentType: false,
    enctype: 'multipart/form-data',
    beforeSend: function () {
      $("#demo-loader").css("display", "block");
      $("#demo-form-caption").empty();
    },
    success: function(data) {
      $("#demo-form-caption").html(data.text);
      $("#demo-form-image").attr("src", data.image);
      $("#demo-loader").css("display", "none");
    },
    error: function(e) {
      console.log(e)
    }
  })
});

function resetState() {
  $("#demo-loader").css("display", "none");
  $("#demo-form-caption").empty();
  $("#upload-loader").css("display", "none");
  $("#upload-form-caption").empty();
  $("#upload-form-image").attr("src", '');
  $("#demo-form-image").attr("src", '');
  $('#upload-image').val('');
}

$("#demo-tab").click(function() {
  resetState();
})

$("#upload-tab").click(function() {
  resetState();
})