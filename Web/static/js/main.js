$(function () {
    $("input[type=file]").change(function () {
        var filename = $(this).val();
        var index = filename.lastIndexOf("\\");
        filename = filename.substring(index + 1, filename.length);
        $(this).parents(".uploader").find(".filename").val(filename);
    });
    $("input[type=file]").each(function () {
        if ($(this).val() == "") {
            $(this).parents(".uploader").find(".filename").val("No file selected...");
        }
    });
});

function toValid() {
    let new_file = document.getElementById("new_file").value;
    if (new_file != "") {
        return true;
    } else {
        alert("请选择需要运行的代码");
        return false;
    }
}

