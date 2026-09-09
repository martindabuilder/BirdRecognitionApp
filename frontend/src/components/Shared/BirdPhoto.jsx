import fallBackImage from "../../assets/fallbackimage.jpg"

import "./bird_photo.css"

function BirdPhoto({ commonName, photo }) {

    if (!photo) {
        return (<img className = "bird-photo" src = {fallBackImage} alt = "No bird photo available"/>)
    }

    return (<img className = "bird-photo" src = {photo} alt = {commonName}/>)
}

export default BirdPhoto