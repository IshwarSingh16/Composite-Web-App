// 8x8 array for composite representation 
const m = 8;
const n = 8;
let arr = [];
for (var i = 0; i < m; i++) {
  arr[i] = [0, 0, 0, 0, 0, 0, 0, 0];
}

var table = document.getElementById("table"), rIndex, cIndex;

for (var i = 0; i < table.rows.length; i++) {
  for (var j = 0; j < table.rows[i].cells.length; j++) {
    table.rows[i].cells[j].onclick = function () {

      rIndex = this.parentElement.rowIndex;
      cIndex = this.cellIndex;

      if (arr[rIndex][cIndex] == 1) { this.style.backgroundColor = 'azure'; arr[rIndex][cIndex] = 0; }

      else {
        this.style.backgroundColor = 'black';
        // console.log("Row: " + rIndex + ", Column:" + cIndex);    
        arr[rIndex][cIndex] = 1;
        // console.log(arr[rIndex][cIndex]);
        // for (var i = 0; i < m; i++) {
        //   console.log(arr[i]); 
        // }
      }
    };
  }
}




