import { BrowserRouter, Routes, Route } from "react-router-dom"

import "./index.css"

import MainSection from "./components/MainSection/MainSection/Main_Section.jsx"
import ResultsSection from "./components/PredictResultsSection/Results_Section.jsx"

function App() {
    return (
        <BrowserRouter>
            <Routes>

                <Route path="/" element={<MainSection />}/>
                <Route path="/results" element={<ResultsSection />}/>

            </Routes>
        </BrowserRouter>
    )
}

export default App