console.log("js import funciona")

function toggleCell(i, j){
    console.log('Cell clicked:', i, j);
    
    // Buscar el componente de texto para cell_data
    const textInputs = document.querySelector("#cell_data > label > div > textarea");

    const data = JSON.stringify({i: parseInt(i), j: parseInt(j)});
    textInputs.value = data;
        
    // Disparar el evento de cambio
    const event = new Event('input', { bubbles: true });
    textInputs.dispatchEvent(event);
}